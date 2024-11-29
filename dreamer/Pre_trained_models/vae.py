import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Encoder(nn.Module):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__()
        self.config = config
        input_dim = self.config.input_dim

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
            self.config.pre_trained_path = kwargs.get('path')

    def forward(self, x):
        # Get encoder output
        latent_params = self.encoder(x)
        mean, log_var = torch.chunk(latent_params, 2, dim=-1)

        log_var = torch.clamp(log_var, min=-10, max=10) # for avoid nan value return and numerical stability
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)

        return dist.rsample(), dist
    

    def save(self, model_name="pre_trained_encoder"):
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        print(f"model saved at {self.config.pre_trained_path}")

    def load(self, model_name="pre_trained_encoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        print(f"model loaded from {self.config.pre_trained_path}")


        

class Decoder(nn.Module):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__()
        self.config = config
        input_dim = self.config.latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim*4),
            nn.ReLU(),
            nn.Linear(input_dim*4, input_dim*8),
            nn.ReLU(),
            nn.Linear(input_dim*8, self.config.input_dim)  # Outputs back to original input size
        )

        if kwargs.get('path'):
            self.config.pre_trained_path = kwargs.get('path')


    def forward(self, z, h):
        x = torch.cat((z, h), dim=-1)
        actual = self.decoder(x)
        return actual
    
    def save(self, model_name="pre_trained_decoder"):
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        print(f"model saved at {self.config.pre_trained_path}")

    def load(self, model_name="pre_trained_decoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        print(f"model loaded from {self.config.pre_trained_path}")



class VAE(nn.Module):
    def __init__(self, config, **kwargs):
        super(VAE, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.config = config

        if kwargs.get('path'):
            self.config.pre_trained_path = kwargs.get('path')

    def forward(self, x, h):
        # Pass through encoder to get latent representation
        z, dist = self.encoder(x)
        
        # Reconstruct through decoder
        reconstructed = self.decoder(z, h)
        
        return reconstructed, dist

    def save(self, model_name="vae"):
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        self.encoder.save()
        self.decoder.save()

    def load(self, model_name="vae"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        self.encoder.load()
        self.decoder.load()




