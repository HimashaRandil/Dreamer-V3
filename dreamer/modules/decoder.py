import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = nn.Sequential(
            nn.Linear(self.config.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.config.input_dim)  # Outputs back to original input size
        )


    def forward(self, x):
        actual = self.decoder(x)
        return actual