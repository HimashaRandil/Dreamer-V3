import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.latent_dim)  # Outputs to the latent space
        )

    def forward(self, x):
        latent = self.encoder(x)
        return latent