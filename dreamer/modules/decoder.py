import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
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
    
    def save(self, model_name="decoder"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="decoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.path, model_name)))