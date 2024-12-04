import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

from dreamer.modules.worldModel import WorldModel
from dreamer.modules.actor_critic import ActorCritic

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

    def imagine_trajectory(self,initial_state):

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

        # Get initial z0 using encoder and initialize h0
        current_z, _ = self.world_model.rssm.e_model(initial_state)  # Initial latent state from encoder
        current_h, _ = self.world_model.rssm.recurrent_model_input_init(batch=1)  # Initial hidden state

        
        for t in range(self.config.horizon):
            # Store current states
            latent_states.append(current_z)
            hidden_states.append(current_h)
            
            # Create state representation for actor by concatenating zt and ht
            state_repr = torch.cat([current_z, current_h], dim=-1)
            
            # Get action from actor network
            action,_ = self.actor_critic.actor.act(state=state_repr)
            actions.append(action)
            
            # Use RSSM to predict next states and outcomes
            # Note: Using your RSSM's transition models
            next_h = self.world_model.rssm.r_model(current_z, self.actor_critic.one_hot_encode(action), current_h)
            _, next_z_prior = self.world_model.rssm.d_model(next_h)
            
            # Predict reward and continue signal using world model components
            reward = self.world_model.reward_predictor(next_z_prior, next_h)  # Using mean of prior
            cont = self.world_model.continue_predictor(next_z_prior, next_h)
            
            rewards.append(reward)
            continues.append(cont)
            
            # Update states for next step
            current_z = next_z_prior  # Use mean of prior distribution
            current_h = next_h
            
                   
        return latent_states, hidden_states, actions, rewards, continues
    
    def train_step(self, initial_state: torch.Tensor) -> Dict[str, float]:
        """Perform one training step using imagined trajectories."""
        # Generate imagined trajectory
        states, actions, rewards, dones = self.imagine_trajectory(initial_state, self.actor_critic.actor, self.config.horizon)
        
        # Get values from critic
        values, _ = self.actor_critic.critic(states, actions)
        
        # Compute λ-returns
        lambda_returns = self.actor_critic.compute_lambda_returns(rewards, values, dones)
        
        # Update networks
        critic_loss = self.actor_critic.update_critic(states, lambda_returns)
        actor_loss = self.actor_critic.update_actor(states, actions, lambda_returns)
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'mean_return': lambda_returns.mean().item(),
            'return_std': lambda_returns.std().item()
        }