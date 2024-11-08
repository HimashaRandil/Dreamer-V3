import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        latent_dim = self.config.latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim*4),
            nn.ReLU(),
            nn.Linear(latent_dim*4, latent_dim*8),
            nn.ReLU(),
            nn.Linear(latent_dim*8, self.config.input_dim)  # Outputs back to original input size
        )


    def forward(self, x):
        actual = self.decoder(x)
        return actual
    
    def save(self, model_name="decoder"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="decoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.path, model_name)))