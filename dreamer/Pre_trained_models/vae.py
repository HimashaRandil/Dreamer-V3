import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Encoder(nn.Module):
    def __init__(self, config, device, **kwargs):
        super(Encoder, self).__init__()
        self.config = config
        input_size = self.config.input_dim
        hidden_dim = self.config.hidden_dim
        input_dim = input_size + hidden_dim
        latent_dim = self.config.latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.ReLU()
        )

        self.device = device

        self.mu = nn.Linear(input_dim // 8, latent_dim)
        self.logvar = nn.Linear(input_dim // 8, latent_dim)

        if kwargs.get('path'):
            self.config.pre_trained_path = kwargs.get('path')

    def forward(self, x):
        h = torch.zeros(self.config.batch_size, self.config.hidden_dim, device=self.device)
        x = torch.cat([x, h], dim=-1)

        e_x = self.encoder(x)
        mu = self.mu(e_x)
        logvar = self.logvar(e_x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return std * eps + mu, logvar, mu
    

    def save(self, model_name="pre_trained_encoder"):
        os.makedirs(self.config.pre_trained_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        print(f"model saved at {self.config.pre_trained_path}")

    def load(self, model_name="pre_trained_encoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        print(f"model loaded from {self.config.pre_trained_path}")

   
    


class Decoder(nn.Module):
    def __init__(self, config, device, **kwargs):
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
            self.config.pre_trained_path = kwargs.get('path')

        self.device = device

    def forward(self, z):
        h = torch.randn(self.config.batch_size, self.config.hidden_dim, device=self.device)
        x = torch.cat((z, h), dim=-1)
        actual = self.decoder(x)
        return torch.sigmoid(actual)
    
     
    def save(self, model_name="pre_trained_decoder"):
        os.makedirs(self.config.pre_trained_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        print(f"model saved at {self.config.pre_trained_path}")

    def load(self, model_name="pre_trained_decoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        print(f"model loaded from {self.config.pre_trained_path}")



class VAE(nn.Module):
    def __init__(self, config, device, **kwargs):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Encoder(config, device=self.device)
        self.decoder = Decoder(config, device=self.device)
        self.config = config

        if kwargs.get('path'):
            self.config.pre_trained_path = kwargs.get('path')

    def forward(self, x):
        # Pass through encoder to get latent representation
        z, logvar, mu = self.encoder(x)
        
        # Reconstruct through decoder
        reconstructed = self.decoder(z)
        
        return reconstructed, logvar, mu

    def save(self, model_name="vae"):
        os.makedirs(self.config.pre_trained_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        self.encoder.save()
        self.decoder.save()

    def load(self, model_name="vae"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        self.encoder.load()
        self.decoder.load()




def vae_loss_function(reconstructed, original, logvar, mu):
    """Calculate VAE loss: Reconstruction Loss + KL Divergence."""
    reconstruction_loss = nn.MSELoss()(reconstructed, original)  # L2 loss
    # KL Divergence: Measure divergence between approximate posterior and prior.
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    return reconstruction_loss + kl_divergence




def train_vae(vae, data_loader, epochs, lr=1e-3, device='cuda'):
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.to(device)

    best_loss = float('inf')  

    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        for loop_count, (obs, rewards, actions, dones, next_obs) in enumerate(data_loader, start=1):  
            obs = obs.to(vae.device)
            # Assume batch is a tuple (input_data, auxiliary_hidden_state)

            # Forward pass
            reconstructed, logvar, mu = vae(obs)

            # Compute loss
            loss = vae_loss_function(reconstructed, obs, logvar, mu)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save the model if it improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            vae.save()
            print(f"New best model saved with loss: {best_loss:.4f}")

    print("Training Complete")

