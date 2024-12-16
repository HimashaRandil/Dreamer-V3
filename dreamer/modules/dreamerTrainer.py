import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

from dreamer.modules.worldModel import WorldModel
from dreamer.modules.actor_critic import ActorCritic
from dreamer.Utils.replayBuffer import ReplayBuffer

from dreamer.Utils.logger import logging

import grid2op
from grid2op.Reward import L2RPNSandBoxScore
from lightsim2grid import LightSimBackend




class DreamerTrainer:
    def __init__(
        self,
        world_model: WorldModel,
        actor_critic: ActorCritic,
        # critic: nn.Module,
        config,
    ):
        self.config =  config 
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = grid2op.make(
            self.config.env_name,  
            reward_class=L2RPNSandBoxScore,
            backend=LightSimBackend()
        )
        logging.info(f"{self.__class__.__name__}.{__name__}: Environment Initialization was sucessful")
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=3200,  # Adjust based on your needs
            sequence_length=self.config.horizon
        )


    def imagine_trajectory(self,initial_state:torch.tensor):

        """
        Generate an imagined trajectory using the trained world model and actor.
        
        Args:
            initial_state: Initial observation tensor
            actor_network: Policy network that generates actions
            horizon: Number of steps to imagine
            
        Returns:
            Tuple of tensors (states, actions, rewards, dones) for the imagined trajectory

        """
        # Initialize storage for trajectory components
        latent_states = []  # store zt
        hidden_states = []  # store ht
        actions = []       # store at
        rewards = []       # store rt
        continues = []     # store ct

        with torch.no_grad():


            current_h, _ = self.world_model.rssm.recurrent_model_input_init(batch=1)  # Initial hidden state
            print(current_h.shape)

            initial_state = initial_state.unsqueeze(0).to(self.world_model.device)
            print(initial_state.shape)
            print(current_h.shape)
            initial_state = torch.cat([initial_state,current_h],dim=-1)
                
            # print(initial_state.shape)
            current_z, _ = self.world_model.rssm.e_model(initial_state)
            print(current_z.shape)
            
            for t in range(self.config.horizon):
                # Store current states
                latent_states.append(current_z.detach())
                hidden_states.append(current_h.detach())
                
                # Create state representation for actor by concatenating zt and ht
                state_repr = torch.cat([current_z, current_h], dim=-1)
                
                # Get action from actor network
                action,_ = self.actor_critic.actor.act(state=state_repr)
                actions.append(action.detach())
                
                # Use RSSM to predict next states and outcomes
                # Note: Using your RSSM's transition models
                next_h = self.world_model.rssm.r_model(current_z, self.actor_critic.one_hot_encode(action), current_h)
                _, next_z_prior = self.world_model.rssm.d_model(next_h)
                
                # Predict reward and continue signal using world model components
                reward = self.world_model.reward_predictor(next_z_prior, next_h)  # Using mean of prior
                cont = self.world_model.continue_predictor(next_z_prior, next_h)
                
                rewards.append(reward.sample().detach())
                continues.append(cont.detach())
                
                # Update states for next step
                current_z = next_z_prior  # Use mean of prior distribution
                current_h = next_h
                 
        return latent_states, hidden_states, actions, rewards, continues
    

    def train_step(self,initial_state: torch.Tensor, replay_buffer_batch: Dict) -> Dict[str, float]:
        """Perform one training step using imagined trajectories."""

        latent_states, hidden_states, actions, rewards, continues = self.imagine_trajectory(
            initial_state=initial_state
        )
        states = torch.stack([torch.cat([z.clone().detach(), h.clone().detach()], dim=-1) for z, h in zip(latent_states, hidden_states)])
        actions = torch.stack([self.actor_critic.one_hot_encode(a).clone().detach() for a in actions])
        rewards = torch.stack([r.clone().detach() for r in rewards])
        continues = torch.stack([c.clone().detach() for c in continues])

        values = []

        for t in range(states.size(0)):
            print("shape of State: ", states[t].shape)
            print("shape of Actiion: ", actions[t].shape)
            value, _ = self.actor_critic.critic(states[t], actions[t])
            values.append(value)
        values = torch.stack(values)
        print("shape of Values", values.shape)

        # Compute λ-returns
        lambda_returns = self.actor_critic.compute_lambda_returns(rewards, values, continues)
        print("shape of lambda Returns ", lambda_returns.shape)

        # Update from imagined trajectory (scale = 1.0)
        critic_loss_imagine = self.actor_critic.update_critic(states, actions, lambda_returns)
        print("Critic_loss_imagine: ", critic_loss_imagine)

        # Get replay buffer data and compute returns
        batch_size = replay_buffer_batch['states'].size(0)  # 32
        seq_len = replay_buffer_batch['states'].size(1)     # 16

        # Reshape to combine batch and sequence dimensions
        replay_states = replay_buffer_batch['states'].view(batch_size * seq_len, -1)  # [32*16, 192]
        replay_actions = replay_buffer_batch['actions'].view(batch_size * seq_len, -1)  # [32*16, action_dim]
        replay_actions = self.actor_critic.one_hot_encode(replay_actions.squeeze(-1))   # [32*16, 179]


        # Get values for all state-action pairs
        replay_values, _ = self.actor_critic.critic(
        replay_states, 
        replay_actions
        )
        print('Replay Values : ' , replay_values)

        
        # Reshape values back to [batch_size, seq_len, 1]
        replay_values = replay_values.view(batch_size, seq_len, -1)

        # Similarly reshape rewards and continues
        replay_rewards = replay_buffer_batch['rewards'].view(batch_size, seq_len, -1)
        replay_continues = replay_buffer_batch['continues'].view(batch_size, seq_len, -1)

        # Compute lambda returns
        replay_lambda_returns = self.actor_critic.compute_lambda_returns(
        replay_rewards,
        replay_values,
        replay_continues
        )

        # For the update, reshape states and actions again
        critic_loss_replay = self.actor_critic.update_critic_replay(
        replay_states,
        replay_actions,
        replay_lambda_returns.view(-1, 1)  # Flatten lambda returns to match states/actions
        )
        print("Replay Critic Loss : ", critic_loss_replay)
        
        # Update actor using imagined trajectories
        actor_loss = self.actor_critic.update_actor(states, actions, lambda_returns)
        print("Actor Loss : ", actor_loss)

        return {
            'actor_loss': actor_loss,
            'critic_loss_imagine': critic_loss_imagine,
            'critic_loss_replay': critic_loss_replay,
            'mean_return': lambda_returns.mean().item(),
        }
    

    def train(self, num_epochs:int, batch_size:int, steps_per_epoch:int): # can transfer these to config
        
        """
        Train the actor-critic using a trained world model.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            steps_per_epoch: Number of training steps per epoch
        """

        self.world_model.load_world_model()
        self.replay_buffer = self.initialize_buffer(num_trajectories=3200)


        # Training metrics
        metrics = {
            'actor_losses': [],
            'critic_imagine_losses': [],
            'critic_replay_losses': [],
            'mean_returns': []
        }
        logging.info(f"{self.__class__.__name__}.{__name__}: Starting actor-critic training...")

        for epoch in range(num_epochs):
            # Initialize epoch metrics
            epoch_metrics = {
                'actor_losses': [],
                'critic_imagine_losses': [],
                'critic_replay_losses': [],
                'mean_returns': []
            }
            
            # Steps within each epoch
            for step in range(steps_per_epoch):
                # Sample batch from replay buffer
                batch = self.replay_buffer.get_sample(batch_size, device=self.device)

                
                # Get initial observation from Grid2Op
                obs = self.env.reset()  # This gives us a Grid2Op observation
                if obs is None:
                    raise RuntimeError("Environment reset failed")
                obs = obs.to_vect()
                obs = torch.tensor(obs)

                # Perform training step
                step_results = self.train_step(
                    initial_state=obs,
                    replay_buffer_batch=batch
                )

                # Add new trajectory occasionally (every 10 steps)
                if step % 10 == 0:
                    self.initialize_buffer(num_trajectories=1)  # Add one new trajectory
                    logging.info(f"Added new trajectory at step {step}. Buffer size: {len(self.replay_buffer)}")

                # Collect metrics
                for key in epoch_metrics:
                    epoch_metrics[key].append(step_results[key.replace('losses', 'loss')])

            # Compute epoch averages
            epoch_mean_metrics = {
                key: np.mean(values) for key, values in epoch_metrics.items()
            }
                
            # Store metrics
            for key in metrics:
                metrics[key].append(epoch_mean_metrics[key])


            # Log progress every N epochs
            if epoch % 10 == 0:
                logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
                logging.info(f"Actor Loss: {epoch_mean_metrics['actor_losses']:.4f}")
                logging.info(f"Critic Imagine Loss: {epoch_mean_metrics['critic_imagine_losses']:.4f}")
                logging.info(f"Critic Replay Loss: {epoch_mean_metrics['critic_replay_losses']:.4f}")
                logging.info(f"Mean Return: {epoch_mean_metrics['mean_returns']:.4f}")
                
            # Save checkpoints periodically
            if epoch % 100 == 0:
                self.save_checkpoint(epoch)
            
        return metrics
    
        
    
    def initialize_buffer(self, num_trajectories: int):
        """
        Initialize the buffer with trajectories generated from the world model.
        
        Args:
            world_model: Trained world model
            env: Grid2Op environment instance
            num_trajectories: Number of trajectories to generate
        """

        # logging.info(f"Generating {num_trajectories} initial trajectories...")

        for i in range(num_trajectories):
            # if i % 10 == 0:
                # logging.info(f"Generated {i}/{num_trajectories} trajectories")
            try: 
                # Get initial observation from Grid2Op
                obs = self.env.reset()  # This gives us a Grid2Op observation
                if obs is None:
                    raise RuntimeError("Environment reset failed")
                obs = obs.to_vect()
                obs = torch.tensor(obs)
                latent_states, hidden_states, actions, rewards, continues = self.imagine_trajectory(
                initial_state=obs
                )
                # Stack all tensors
                trajectory = {
                    'states': torch.stack([torch.cat([z.clone().detach(), h.clone().detach()], dim=-1) 
                                        for z, h in zip(latent_states, hidden_states)]),
                    'actions': torch.stack([a.clone().detach() for a in actions]),
                    'rewards': torch.stack([r.clone().detach() for r in rewards]),
                    'continues': torch.stack([c.clone().detach() for c in continues])
                }
                self.replay_buffer.add(trajectory)
            except Exception as e:
                raise RuntimeError(f"Sequence generation failed: {e}")
        logging.info(f"{self.__class__.__name__}.{__name__}: Buffer Size {self.replay_buffer.__len__()} initialization complete!")


    

    def save_checkpoint(self, epoch):
        """Save actor-critic checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor_critic.actor.state_dict(),
            'critic_state_dict': self.actor_critic.critic.state_dict(),
            'critic_target_state_dict': self.actor_critic.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_critic.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.actor_critic.critic_optimizer.state_dict(),
            'epoch': epoch
            }
        torch.save(checkpoint, f'actor_critic_checkpoint_{epoch}.pt')
