import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ContinuousPredictor(nn.Module):
    def __init__(self, config, **kwargs):
        super(ContinuousPredictor, self).__init__()

        self.config = config
        self.network = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        if kwargs.get('path'):
            self.config.path = kwargs.get('path')


    def forward(self, latent_dim, hidden_dim, threshold=False):
        x = torch.cat((latent_dim, hidden_dim), dim=-1)
        x = self.network(x)
        
        #dist = torch.distributions.Bernoulli(logits=x)   #Bernoulli distribution is a natural choice for modeling binary events
        return x
    

    def save(self, model_name="continue_predictor"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="continue_predictor", custom_path=None):
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