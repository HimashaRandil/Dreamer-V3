import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DynamicPredictor(nn.Module):
    def __init__(self, config):
        super(DynamicPredictor, self).__init__()

        self.config = config

        self.config = config
        h = self.config.hidden_dim
        self.network = nn.Sequential(
            nn.Linear(h, h//2),
            nn.ReLU(),
            nn.Linear(h//2, h//4),
            nn.ReLU(),
            nn.Linear(h//4, h//8),
            nn.ReLU(),
            nn.Linear(h//8, self.config.latent_dim)  # Output mean and log-std for latent state distribution
        )

    def forward(self, h_t):
        x = self.network(h_t)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(log_std) + 1e-6
        dist = torch.distributions.Normal(mean, std)
        return dist
    

    def save(self, model_name="dynamic_predictor"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="dynamic_predictor"):
        self.load_state_dict(torch.load(os.path.join(self.config.path, model_name)))