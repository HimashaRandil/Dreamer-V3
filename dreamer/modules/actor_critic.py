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
        return torch.sign(x) * torch.log(1 + torch.abs(x))
    
    def symexp(self, x):
        """Inverse of symlog."""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    
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
        lambda_returns = torch.zeros_like(rewards)
        # Handle the final step: R^λ_T = vT
        lambda_returns[-1] = values[-1]
        
        for t in reversed(range(len(values)- 1)):
            bootstrap = (1 - self.lambda_gae) * values[t] + self.lambda_gae * lambda_returns[t + 1]
            lambda_returns[t] = rewards[t] + self.gamma * continues[t] * bootstrap
            
        return lambda_returns
    
    def update_critic(self, states: torch.Tensor, actions: torch.Tensor, lambda_returns: torch.Tensor) -> float:
        """Update critic using discrete regression with twohot targets."""
        # Transform and encode targets
        transformed_returns = self.symlog(lambda_returns)
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
        transformed_returns = self.symlog(lambda_returns)
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
        # Normalize returns
        if self.return_ema is None:
            self.return_ema = lambda_returns.mean()
            self.return_std = lambda_returns.std()
        else:
            self.return_ema = 0.99 * self.return_ema + 0.01 * lambda_returns.mean()
            self.return_std = 0.99 * self.return_std + 0.01 * lambda_returns.std()
        
        normalized_returns = lambda_returns / torch.max(
            torch.ones_like(lambda_returns),
            lambda_returns.abs() / self.return_std
        )
        
        # Get action distribution
        action_logits = self.actor(states)
        log_probs, entropy, _ = self.actor.evaluate_actions(action_logits, actions)
        
        # Compute actor loss
        policy_loss = -(log_probs * normalized_returns.detach()).mean()
        entropy_loss = -entropy.mean()
        actor_loss = policy_loss - self.entropy_scale * entropy_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    

    def one_hot_encode(self, tensor):
        """
        One-hot encode a tensor of indices.
        
        Args:
            tensor (torch.Tensor): Tensor containing class indices (e.g., tensor([104], device='cuda:0')).
            num_classes (int): Total number of classes.
            
        Returns:
            torch.Tensor: One-hot encoded tensor.
        """
        # Ensure tensor is long type for indexing
        tensor = tensor.long()
        
        # Create a one-hot encoded tensor
        one_hot = torch.zeros(tensor.size(0), self.config.action_dim, device=tensor.device)
        one_hot.scatter_(1, tensor.unsqueeze(1), 1)
        
        return one_hot
