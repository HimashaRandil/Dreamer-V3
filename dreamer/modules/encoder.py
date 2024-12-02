import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class Encoder(nn.Module):
    def __init__(self, config, **kwargs):
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
            nn.Linear(input_dim//8, self.config.latent_dim*2)
        )

        if kwargs.get('path'):
            self.config.path = kwargs.get('path')

    def forward(self, x):
        # Get encoder output
        latent_params = self.encoder(x)
        mean, log_var = torch.chunk(latent_params, 2, dim=-1)

        log_var = torch.clamp(log_var, min=-10, max=10) # for avoid nan value return and numerical stability
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)

        return dist.rsample(), dist
    

    def save(self, model_name="encoder"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="encoder", custom_path=None):
        if custom_path:
            model_path = os.path.join(custom_path, model_name)
        else:
            model_path = os.path.join(self.config.saved_model_path, model_name)
    
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Model file not found at {model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")