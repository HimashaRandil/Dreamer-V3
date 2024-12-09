import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

from dreamer.modules.worldModel import WorldModel
from dreamer.modules.actor_critic import ActorCritic
from dreamer.Utils.replayBuffer import ReplayBuffer




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

    def imagine_trajectory(self,initial_state,is_encoded=False):

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

            # Get initial z0 using encoder and initialize h0
            if not is_encoded:
                current_z, _ = self.world_model.rssm.e_model(initial_state)
            else:
                current_z = initial_state  # Already encoded
            current_h, _ = self.world_model.rssm.recurrent_model_input_init(batch=initial_state.size(0))  # Initial hidden state

            
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
                
                rewards.append(reward.detach())
                continues.append(cont.detach())
                
                # Update states for next step
                current_z = next_z_prior  # Use mean of prior distribution
                current_h = next_h
            
                   
        return latent_states, hidden_states, actions, rewards, continues
    
    def train_step(self, initial_state: torch.Tensor, replay_buffer_batch: Dict) -> Dict[str, float]:
        """Perform one training step using imagined trajectories."""
        # Generate imagined trajectory
        try:

            if initial_state.size(-1) != self.world_model.latent_dim:
                raise ValueError(
                    f"Initial state dimension {initial_state.size(-1)} does not match "
                    f"expected encoded state dimension {self.world_model.latent_dim}. "
                    "Make sure you're passing encoded states from replay buffer."
                )

            latent_states, hidden_states, actions, rewards, continues =self.imagine_trajectory(initial_state=initial_state, is_encoded=True)

            # Stack all latent and hidden states from all timesteps
            latent_stack = torch.stack(latent_states)    # Shape: [time_steps, batch_size, latent_dim]
            hidden_stack = torch.stack(hidden_states)    # Shape: [time_steps, batch_size, hidden_dim]
            
            # Concatenate latent and hidden states
            states = torch.cat([latent_stack, hidden_stack], dim=-1)  # Shape: [time_steps, batch_size, latent_dim + hidden_dim]
            actions = torch.stack(actions)                           # Shape: [time_steps, batch_size, action_dim]
            rewards = torch.stack(rewards)                          # Shape: [time_steps, batch_size]
            continues = torch.stack(continues)

            
            values = [] 
            
            for t in range(states.size(0)):
                value, _ = self.actor_critic.critic(states[t], actions[t])
                values.append(value)
            values = torch.stack(values)

            # Compute λ-returns
            lambda_returns = self.actor_critic.compute_lambda_returns(rewards, values, continues)
            
            # Update from imagined trajectory (scale = 1.0)
            critic_loss_imagine = self.actor_critic.update_critic(states, actions, lambda_returns)
            
            # Get replay buffer data
            replay_states = replay_buffer_batch['states']
            replay_actions = replay_buffer_batch['actions']
            replay_rewards = replay_buffer_batch['rewards']
            replay_continues = replay_buffer_batch['continues']
            
            # Compute returns for replay data
            replay_values, _ = self.actor_critic.critic(replay_states, replay_actions)
            replay_lambda_returns = self.actor_critic.compute_lambda_returns(
                replay_rewards, replay_values, replay_continues
            )
            
            # Update from replay data (scale = 0.3)
            critic_loss_replay = self.actor_critic.update_critic_replay(
                replay_states, replay_actions, replay_lambda_returns
            )
            
            # Rest of the training step...
            actor_loss = self.actor_critic.update_actor(states, actions, lambda_returns)
            
            return {
                'actor_loss': actor_loss,
                'critic_loss_imagine': critic_loss_imagine,
                'critic_loss_replay': critic_loss_replay,
                'mean_return': lambda_returns.mean().item(),
            }

        except Exception as e:
            print(f"Error in train_step: {e}")
            raise


    def train(self, trained_world_model:WorldModel,
              num_epochs:int, batch_size:int, steps_per_epoch:int):
        
        """
        Train the actor-critic using a trained world model.
        
        Args:
            trained_world_model: Pre-trained world model
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            steps_per_epoch: Number of training steps per epoch
        """
        # Import and create Grid2Op environment
        import grid2op
        from grid2op.Reward import L2RPNSandBoxScore
        from lightsim2grid import LightSimBackend

        # Create environment specifically for case 14
        env = grid2op.make(
            "l2rpn_case14_sandbox",  # Specific to case 14
            reward_class=L2RPNSandBoxScore,
            backend=LightSimBackend()
        )
        
        # Make sure world model is in eval mode
        trained_world_model.eval()
        
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(
            capacity=1000,  # Adjust based on your needs
            actor_critic=self.actor_critic,
            sequence_length=self.config.horizon,
            device=self.device
        )
        
        # Initialize buffer with imagined trajectories
        # Initialize buffer using the existing method
        replay_buffer.initialize_buffer(
            world_model=trained_world_model,env=env,
            num_trajectories=100
        )
        
        # Training metrics
        metrics = {
            'actor_losses': [],
            'critic_imagine_losses': [],
            'critic_replay_losses': [],
            'mean_returns': []
        }
        
        print("Starting actor-critic training...")
        for epoch in range(num_epochs):
            epoch_metrics = {
                'actor_losses': [],
                'critic_imagine_losses': [],
                'critic_replay_losses': [],
                'mean_returns': []
            }
            
            for step in range(steps_per_epoch):
                # Sample batch from replay buffer
                batch = replay_buffer.sample(batch_size)
                
                # Get initial state from batch
                initial_state = batch['states'][:, 0, :]  # [batch_size, obs_dim]
                
                # Perform training step
                step_results = self.train_step(
                    initial_state=initial_state,
                    replay_buffer_batch={
                        'states': batch['states'],
                        'actions': batch['actions'],
                        'rewards': batch['rewards'],
                        'continues': batch['continues']
                    }
                )
                
                # Collect metrics
                for key in epoch_metrics:
                    epoch_metrics[key].append(step_results[key.replace('losses', 'loss')])
                
                # Generate new trajectory and add to buffer occasionally
                if step % 10 == 0:  # Adjust frequency as needed
                    new_trajectory = replay_buffer.generate_sequence(trained_world_model,env)
                    replay_buffer.add(new_trajectory)
            
            # Compute epoch averages
            epoch_mean_metrics = {
                key: np.mean(values) for key, values in epoch_metrics.items()
            }
            
            # Store metrics
            for key in metrics:
                metrics[key].append(epoch_mean_metrics[key])
            
            # Print progress
            if epoch % 10 == 0:  # Print every 10 epochs
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Actor Loss: {epoch_mean_metrics['actor_losses']:.4f}")
                print(f"Critic Imagine Loss: {epoch_mean_metrics['critic_imagine_losses']:.4f}")
                print(f"Critic Replay Loss: {epoch_mean_metrics['critic_replay_losses']:.4f}")
                print(f"Mean Return: {epoch_mean_metrics['mean_returns']:.4f}")
            
            # Save checkpoints periodically
            if epoch % 100 == 0:
                self.save_checkpoint(epoch)
        
        return metrics


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
