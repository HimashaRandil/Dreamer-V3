import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = self.config.input_dim
        hidden_dim = self.config.hidden_dim
        input_dim = input_size + hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, input_dim//8),
            nn.ReLU(),
            nn.Linear(input_dim//8, input_dim//16),
            nn.ReLU(),
            nn.Linear(input_dim//16, self.config.latent_dim *2)  # Outputs to the latent space
        )

    def forward(self, x):
        latent_params = self.encoder(x)
        mean, log_var = torch.chunk(latent_params, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()  # rsample allows gradients to flow through the sample
        return z, dist
    

    def save(self, model_name="encoder"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="encoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.path, model_name)))