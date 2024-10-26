import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
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
    

    def save(self, model_name="encoder"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="encoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.path, model_name)))