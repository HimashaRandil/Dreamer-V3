import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

from dreamer.modules.worldModel import WorldModel

from dreamer.modules.networks import ActorNetwork, CriticNetwork


''' ----- Code Under Development----'''


class ActorCritic:
    def __init__(
        self,
        config,
        device: str = "cuda"
    ):
        self.config =  config 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(
            in_dim=self.config.latent_dim + self.config.hidden_dim,
            action_dim=self.config.action_dim,
            actor_hidden_dims=self.config.actor_hidden_dims,
            layer_norm=True,
            activation=nn.ELU,                  # default from ActorNetwork
            epsilon=self.config.actor_epsilon 
        ).to(device)

        self.critic = CriticNetwork(
            obs_dim=config.latent_dim + config.hidden_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.critic_hidden_dims,
            num_buckets=self.config.num_buckets
        ).to(device)

        self.critic_target= CriticNetwork(
            obs_dim=config.latent_dim + config.hidden_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.critic_hidden_dims,
            num_buckets=self.config.num_buckets
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.beta_val = self.config.beta_val
        self.beta_repval = self.config.beta_repval

        self.horizon = self.config.horizon
        self.gamma = self.config.gamma
        self.lambda_gae = self.config.lambda_gae
        self.entropy_scale = self.config.entropy_scale
        self.critic_ema_decay = self.config.critic_ema_decay
        
        # Setup value discretization
        self.num_buckets = self.config.num_buckets
        # self.bucket_values = torch.linspace(-20, 20, self.num_buckets).to(device) 
        # Exponentially SPaced Bins used in Dreamer-V3 version 2
        self.bucket_values = torch.sign(torch.linspace(-20, 20, self.num_buckets)) * (torch.exp(torch.abs(torch.linspace(-20, 20, self.num_buckets))) - 1)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_critic_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.actor_critic_lr)
        
        # Running statistics for return normalization
        self.return_ema = None
        self.return_std = None
        
    def symlog(self, x):
        """Symmetric log transformation."""
        # Clip input to prevent infinite values
        x = torch.clamp(x, -1e6, 1e6)
        return torch.sign(x) * torch.log(1 + torch.clamp(torch.abs(x), min=1e-6))
    
    def symexp(self, x):
        """Inverse of symlog."""
        # Clip input to prevent infinite values
        x = torch.clamp(x, -1e6, 1e6)
        return torch.sign(x) * (torch.exp(torch.clamp(torch.abs(x), max=15)) - 1)
    
    def twohot_encode(self, x):
        """Compute twohot encoding."""
        x_norm = (x + 20) * (self.num_buckets - 1) / 40  # Normalize to [0, num_buckets-1]
        lower_idx = torch.floor(x_norm).long()
        upper_idx = torch.ceil(x_norm).long()
        
        lower_weight = upper_idx.float() - x_norm
        upper_weight = 1 - lower_weight
        
        encoding = torch.zeros(*x.shape, self.num_buckets, device=x.device)
        encoding.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        encoding.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
        
        return encoding
    
    def compute_lambda_returns(
        self, 
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute λ-returns as described in the paper.
        R^λ_t = rt + γct[(1-λ)vt + λR^λ_{t+1}]
        """
        if not (rewards.shape == values.shape == continues.shape):
            raise ValueError(f"Shape mismatch: rewards {rewards.shape}, values {values.shape}, "
                           f"continues {continues.shape}")
        
        # Scale rewards to prevent extreme values
        reward_scale = rewards.abs().mean().item()
        if reward_scale > 1:
            rewards = rewards / reward_scale
            values = values / reward_scale


        
        lambda_returns = torch.zeros_like(rewards)
        # Handle the final step: R^λ_T = vT
        lambda_returns[-1] = values[-1]
        
        for t in reversed(range(len(values)- 1)):
            bootstrap = (1 - self.lambda_gae) * values[t] + self.lambda_gae * lambda_returns[t + 1]
            lambda_returns[t] = rewards[t] + self.gamma * continues[t] * bootstrap

        # Rescale returns if we scaled rewards
        if reward_scale > 1:
            lambda_returns = lambda_returns * reward_scale
            
        return lambda_returns
   
    
    def update_critic(self, states: torch.Tensor, actions: torch.Tensor, lambda_returns: torch.Tensor) -> float:
        """Update critic using discrete regression with twohot targets."""
        # print("States Shape in update critic", states.shape)

        # Transform and encode targets
        transformed_returns = self.symlog(lambda_returns.clamp(-1e6, 1e6))
        target_distribution = self.twohot_encode(transformed_returns)
        
        # Get critic predictions
        value_logits = self.critic.get_value_distribution(states, actions)  # Need to modify critic
        value_distribution = F.softmax(value_logits, dim=-1)
        
        # Compute loss (categorical cross-entropy)
        critic_loss = -(target_distribution * torch.log(value_distribution + 1e-8)).sum(-1).mean()
        
        # Update critic with full scale (beta_val = 1.0)
        loss = critic_loss * self.beta_val
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # Add gradient clipping
        self.critic_optimizer.step()

        # Update target network
        self._update_target_network()
        
        return critic_loss.item()
    
    def update_critic_replay(self, states: torch.Tensor, actions: torch.Tensor, lambda_returns: torch.Tensor) -> float:
        """
        Update critic using replay buffer trajectories (scale = 0.3)
        
        This is important because:
        1. Helps ground predictions in real experiences
        2. Prevents divergence from actual environment dynamics
        3. Balances between imagination and reality
        
        Args:
            states: Real states from replay buffer
            actions: Real actions from replay buffer
            lambda_returns: Computed returns from real trajectories
        """
        transformed_returns = self.symlog(lambda_returns.clamp(-1e6, 1e6))
        target_distribution = self.twohot_encode(transformed_returns)
        
        value_logits = self.critic.get_value_distribution(states, actions)
        value_distribution = F.softmax(value_logits, dim=-1)
        
        critic_loss = -(target_distribution * torch.log(value_distribution + 1e-8)).sum(-1).mean()
        
        # Update critic with reduced scale (beta_repval = 0.3)
        loss = critic_loss * self.beta_repval
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        return loss.item()
    

    def _update_target_network(self):
        """Helper method for target network updates"""
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), 
                                         self.critic_target.parameters()):
                target_param.data.copy_(
                    self.critic_ema_decay * target_param.data + 
                    (1 - self.critic_ema_decay) * param.data
                )
    
    
    def update_actor(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lambda_returns: torch.Tensor
    ) -> float:
        """Update actor using normalized returns and entropy regularization."""
        """Update actor with robust normalization."""
        # Get value estimates for advantage computation
        values, _ = self.critic(states, actions)
        values = self.symexp(values)  # Convert back from symlog space
        
        # Compute advantages
        advantages = lambda_returns - values.detach()
        
        # Robust normalization
        if self.return_ema is None:
            self.return_ema = advantages.mean()
            self.return_std = advantages.std().clamp(min=1.0)
        else:
            self.return_ema = 0.99 * self.return_ema + 0.01 * advantages.mean()
            self.return_std = (0.99 * self.return_std + 0.01 * advantages.std()).clamp(min=1.0)
        
        # Normalize advantages with clipping
        normalized_advantages = (advantages - self.return_ema) / self.return_std
        normalized_advantages = normalized_advantages.clamp(-10.0, 10.0)
        
        # Get action distribution and compute loss
        action_logits = self.actor(states)
        log_probs, entropy, _ = self.actor.evaluate_actions(action_logits, actions)
        
        # Compute actor loss with clipped advantages
        policy_loss = -(log_probs * normalized_advantages.detach()).mean()
        entropy_loss = -entropy.mean()
        actor_loss = policy_loss - self.entropy_scale * entropy_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # Add gradient clipping
        self.actor_optimizer.step()
        
        return actor_loss.item()
    

    def one_hot_encode(self, tensor):
        """
        One-hot encode a tensor of action indices.
        
        Args:
            tensor (torch.Tensor): Action indices tensor. Can be:
                - 0D: single action (scalar)
                - 1D: [batch_size] multiple actions
                - 2D: [batch_size, 1] multiple actions
                
        Returns:
            torch.Tensor: One-hot encoded tensor of shape [batch_size, action_dim]
        """
        # Convert to long dtype for indexing
        tensor = tensor.long()
        
        # Handle different input dimensions
        if tensor.dim() == 0:  # Single action (scalar)
            tensor = tensor.unsqueeze(0)  # Add batch dimension [1]
        elif tensor.dim() == 2:  # Already has second dimension [batch_size, 1]
            tensor = tensor.squeeze(-1)  # Remove second dimension
        # tensor.dim() == 1 case doesn't need modification
        
        # Create zero tensor with proper batch size
        batch_size = tensor.size(0)
        one_hot = torch.zeros(batch_size, self.config.action_dim, device=tensor.device)
        
        # Perform one-hot encoding
        one_hot.scatter_(1, tensor.unsqueeze(1), 1)
        
        return one_hot
