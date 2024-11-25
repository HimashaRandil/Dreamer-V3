import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

from networks import ActorNetwork, CriticNetwork


''' ----- Code Under Development----'''


class DreamerTrainer:
    def __init__(
        self,
        world_model: nn.Module,
        actor: ActorNetwork,
        critic: CriticNetwork,
        # critic: nn.Module,
        horizon: int = 16,  # T=16 from paper
        gamma: float = 0.997,  # From paper
        lambda_gae: float = 0.95,
        entropy_scale: float = 3e-4,  # η from paper
        critic_ema_decay: float = 0.995,
        num_buckets: int = 255,  # K from paper
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        
        # Initialize critic target network (EMA)
        self.critic_target = type(critic)(*critic.__init_args__).to(device)
        self.critic_target.load_state_dict(critic.state_dict())
        
        self.horizon = horizon
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.entropy_scale = entropy_scale
        self.critic_ema_decay = critic_ema_decay
        self.device = device
        
        # Setup value discretization
        self.num_buckets = num_buckets
        self.bucket_values = torch.linspace(-20, 20, num_buckets).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
        
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
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute λ-returns as described in the paper."""
        lambda_returns = torch.zeros_like(rewards)
        next_value = values[-1]
        
        for t in reversed(range(self.horizon)):
            bootstrap = (1 - self.lambda_gae) * values[t + 1] + self.lambda_gae * next_value
            lambda_returns[t] = rewards[t] + self.gamma * (1 - dones[t]) * bootstrap
            next_value = lambda_returns[t]
            
        return lambda_returns
    
    def update_critic(self, states: torch.Tensor, lambda_returns: torch.Tensor) -> float:
        """Update critic using discrete regression with twohot targets."""
        # Transform and encode targets
        transformed_returns = self.symlog(lambda_returns)
        target_distribution = self.twohot_encode(transformed_returns)
        
        # Get critic predictions
        value_logits = self.critic.get_value_distribution(states)  # Need to modify critic
        value_distribution = F.softmax(value_logits, dim=-1)
        
        # Compute loss (categorical cross-entropy)
        critic_loss = -(target_distribution * torch.log(value_distribution + 1e-8)).sum(-1).mean()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update target network (EMA)
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), 
                                         self.critic_target.parameters()):
                target_param.data.copy_(
                    self.critic_ema_decay * target_param.data + 
                    (1 - self.critic_ema_decay) * param.data
                )
        
        return critic_loss.item()
    
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
    
    def train_step(self, initial_state: torch.Tensor) -> Dict[str, float]:
        """Perform one training step using imagined trajectories."""
        # Generate imagined trajectory
        states, actions, rewards, dones = self.world_model.imagine_trajectory(
            initial_state, self.actor, self.horizon
        )
        
        # Get values from critic
        values, _ = self.critic(states, actions)
        
        # Compute λ-returns
        lambda_returns = self.compute_lambda_returns(rewards, values, dones)
        
        # Update networks
        critic_loss = self.update_critic(states, lambda_returns)
        actor_loss = self.update_actor(states, actions, lambda_returns)
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'mean_return': lambda_returns.mean().item(),
            'return_std': lambda_returns.std().item()
        }