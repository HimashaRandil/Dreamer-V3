import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ContinuousPredictor(nn.Module):
    def __init__(self, config):
        super(ContinuousPredictor, self).__init__()

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

        dist = torch.distributions.Bernoulli(logits=x)   #Bernoulli distribution is a natural choice for modeling binary events
        return dist