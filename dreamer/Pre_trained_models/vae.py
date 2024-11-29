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


    def forward(self, x):
        #x = torch.cat((z, h), dim=-1)
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

    def forward(self, x):
        # Pass through encoder to get latent representation
        z, dist = self.encoder(x)
        
        # Reconstruct through decoder
        reconstructed = self.decoder(z)
        
        return reconstructed, dist

    def save(self, model_name="vae"):
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        self.encoder.save()
        self.decoder.save()

    def load(self, model_name="vae"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        self.encoder.load()
        self.decoder.load()




def vae_loss_function(reconstructed, original, dist):
    """Calculate VAE loss: Reconstruction Loss + KL Divergence."""
    reconstruction_loss = nn.MSELoss()(reconstructed, original)  # L2 loss
    mean, std = dist.mean, dist.stddev
    
    # KL Divergence: Measure divergence between approximate posterior and prior.
    kl_divergence = -0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2, dim=-1).mean()

    return reconstruction_loss + kl_divergence




def train_vae(vae, data_loader, epochs, lr=1e-3, device='cuda'):
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.to(device)

    best_loss = float('inf')  

    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        for batch in data_loader:
            # Assume batch is a tuple (input_data, auxiliary_hidden_state)
            x = batch
            x = x.to(device)

            # Forward pass
            reconstructed, dist = vae(x)

            # Compute loss
            loss = vae_loss_function(reconstructed, x, dist)
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
