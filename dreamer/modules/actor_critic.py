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
        # world_model: WorldModel,
        # actor: ActorNetwork,
        # critic: CriticNetwork,
        # critic: nn.Module,
        config,
        device: str = "cuda"
    ):
        self.config =  config 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.world_model = world_model
        # self.actor = actor
        # self.critic = critic
        
        # Initialize critic target network (EMA)
        # self.critic_target = type(critic)(*critic.__init_args__).to(device)

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
            hidden_dim=400,
            num_buckets=self.config.num_buckets
        ).to(device)

        self.critic_target= CriticNetwork(
            obs_dim=config.latent_dim + config.hidden_dim,
            action_dim=self.config.action_dim,
            hidden_dim=400,
            num_buckets=self.config.num_buckets
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        

        self.horizon = self.config.horizon
        self.gamma = self.config.gamma
        self.lambda_gae = self.config.lambda_gae
        self.entropy_scale = self.config.entropy_scale
        self.critic_ema_decay = self.config.critic_ema_decay
        self.device = device
        
        # Setup value discretization
        self.num_buckets = self.config.num_buckets
        self.bucket_values = torch.linspace(-20, 20, self.config.num_buckets).to(device)
        
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
    