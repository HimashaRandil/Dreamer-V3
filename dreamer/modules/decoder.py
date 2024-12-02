import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class Decoder(nn.Module):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__()
        self.config = config
        latent_dim = self.config.latent_dim
        hidden_dim = self.config.hidden_dim
        input_dim = latent_dim + hidden_dim

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, self.config.input_dim)  # Outputs back to original input size
        )

        if kwargs.get('path'):
            self.config.path = kwargs.get('path')


    def forward(self, z, h):
        x = torch.cat((z, h), dim=-1)
        actual = self.decoder(x)
        return actual
    
    def save(self, model_name="decoder"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="decoder", custom_path=None):
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