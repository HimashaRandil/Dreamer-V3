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
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, self.config.latent_dim * 2)  # Output mean and log-variance
        )

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

    def load(self, model_name="dynamic_predictor"):
        self.load_state_dict(torch.load(os.path.join(self.config.path, model_name)))