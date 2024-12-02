import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DynamicPredictor(nn.Module):
    def __init__(self, config, **kwargs):
        super(DynamicPredictor, self).__init__()

        self.config = config

        self.config = config
        h = self.config.hidden_dim

        self.network = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, self.config.latent_dim * 2)  # Output mean and log-variance
        )

        if kwargs.get('path'):
            self.config.path = kwargs.get('path')

    def forward(self, h_t):
        try:
            x = self.network(h_t)
            mean, log_std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(log_std) + 1e-6 # we employ free bits by clipping the dynamics and representation losses below the value of 1 nat ≈ 1.44 bits.
            dist = torch.distributions.Normal(mean, std)
            return dist, dist.rsample()
        except Exception as e:
            print(f"{e}\n\nat Dynamic predictor")
    

    def input_init(self):
        return torch.zeros(self.config.batch_size, self.config.hidden_dim).to(self.config.device)
    

    def save(self, model_name="dynamic_predictor"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="dynamic_predictor", custom_path=None):
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