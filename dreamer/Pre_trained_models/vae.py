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
        self.device = device

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

    def forward(self, x, h):
        # Get encoder output
        #h = torch.randn(self.config.batch_size, self.config.hidden_dim, device=self.device)
        x = torch.cat([h, x], dim=-1)
        latent_params = self.encoder(x)

        mean, log_var = torch.chunk(latent_params, 2, dim=-1)

        log_var = torch.clamp(log_var, min=-10, max=10) # for avoid nan value return and numerical stability
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)

        return dist.rsample(), dist
    

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
        self.device = device

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
    
     
    def save(self, model_name="pre_trained_decoder"):
        os.makedirs(self.config.pre_trained_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        print(f"model saved at {self.config.pre_trained_path}")

    def load(self, model_name="pre_trained_decoder"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        print(f"model loaded from {self.config.pre_trained_path}")



class VAE(nn.Module):
    def __init__(self, config, device):
        super(VAE, self).__init__()
        self.config = config
        self.device = device
        self.encoder = Encoder(config, device=self.device)
        self.decoder = Decoder(config, device=self.device)
        

    def forward(self, x):
        h = torch.zeros(self.config.batch_size, self.config.hidden_dim).to(self.device) 

        z, dist = self.encoder(x, h)
        mean = dist.mean
        log_var = dist.scale.log() ** 2  
        
        reconstructed = self.decoder(z, h)

        return reconstructed, mean, log_var


    def save(self, model_name="vae"):
        os.makedirs(self.config.pre_trained_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.config.pre_trained_path, model_name))
        self.encoder.save()
        self.decoder.save()

    def load(self, model_name="vae"):
        self.load_state_dict(torch.load(os.path.join(self.config.pre_trained_path, model_name)))
        self.encoder.load()
        self.decoder.load()


    def kl_divergence_loss(self, mean, log_var):
        """
        Compute the KL Divergence Loss.
        :param mean: Mean (\mu) of the latent distribution.
        :param log_var: Log variance (\log \sigma^2) of the latent distribution.
        :return: KL Divergence loss (scalar).
        """
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        return kl_loss.mean()
    
    def reconstruction_loss(self, reconstructed, original):
        return nn.MSELoss()(reconstructed, original)
    

    def vae_loss_function(self, reconstructed, original, mean, log_var):
        """
        Compute the total VAE loss: Reconstruction Loss + KL Divergence Loss.
        """
        # Compute individual losses
        rec_loss = self.reconstruction_loss(reconstructed, original)
        kl_loss = self.kl_divergence_loss(mean, log_var)
        
        # Total loss
        return rec_loss + kl_loss







def train_vae(vae:VAE, data_loader, epochs, lr=1e-3, device='cuda'):
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
            reconstructed, mean, log_var = vae(obs)

            # Compute loss
            loss = vae.vae_loss_function(reconstructed, obs, mean, log_var)
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

