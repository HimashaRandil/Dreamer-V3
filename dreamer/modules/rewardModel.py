import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import os

class RewardPredictor(nn.Module):
    def __init__(self, config):
        super(RewardPredictor, self).__init__()

        self.config = config
        self.network = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, latent_dim, hidden_dim):
        x = torch.cat((latent_dim, hidden_dim), dim=-1)
        x = self.network(x)
        
        # Split the output into mean and log-variance
        mean, log_std = x.chunk(2, dim=-1)
        
        # Use softplus to ensure positive std deviation
        std = F.softplus(log_std) + 1e-6  # Add small epsilon for numerical stability
        
        # Create the Normal distribution for stochastic output
        dist = Normal(mean, std)
        
        return dist
    
    def save(self, model_name="reward_predictor"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="reward_predictor"):
        self.load_state_dict(torch.load(os.path.join(self.config.path, model_name)))