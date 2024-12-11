import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dreamer.modules.continueModel import ContinuousPredictor
from dreamer.modules.decoder import Decoder
from dreamer.modules.encoder import Encoder
from dreamer.modules.dynamics import DynamicPredictor
from dreamer.modules.recurrentModel import RecurrentModel
from dreamer.modules.rewardModel import RewardPredictor


class RSSM(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        
        if kwargs.get('path'):
            path = kwargs.get('path') 
        
            self.r_model = RecurrentModel(self.config, path=path)
            self.d_model = DynamicPredictor(self.config, path=path)
            self.e_model = Encoder(self.config, path=path)
            self.decoder = Decoder(self.config, path=path)

        self.r_model = RecurrentModel(self.config)
        self.d_model = DynamicPredictor(self.config)
        self.e_model = Encoder(self.config)
        self.decoder = Decoder(self.config)

        self.optimizer = optim.Adam([
                {'params': self.r_model.parameters()},  
                {'params': self.d_model.parameters()},
                {'params': self.e_model.parameters()},
                {'params': self.decoder.parameters()}
            ], lr=float(self.config.learning_rate))

    def recurrent_model_input_init(self, batch=None):
        if batch:
            return self.r_model.input_init(batch), self.r_model.action_init()
        else:
            return self.r_model.input_init(), self.r_model.action_init()

    
    def forward(self, obs, hidden_state, action):
        # Assert feature dimensions
        assert obs.shape[1] == self.config.input_dim, f"Expected obs feature size {self.config.input_dim}, got {obs.shape[1]}"
        assert hidden_state.shape[1] == self.config.hidden_dim, f"Expected hidden_state size {self.config.hidden_dim}, got {hidden_state.shape[1]}"
        assert action.shape[1] == self.config.action_dim, f"Expected action size {self.config.action_dim}, got {action.shape[1]}"


        x = torch.cat([obs, hidden_state], dim=-1)
        assert x.shape[1] == self.config.input_dim + self.config.hidden_dim

        latent_sample, posterior_dist = self.e_model(x)

        reconstruct_ob = self.decoder(latent_sample, hidden_state)
        
        h = self.r_model(latent_sample, action, hidden_state)
        
        prior_dist, prior_latent_sample = self.d_model(h)
        
        return latent_sample, posterior_dist, reconstruct_ob, h, prior_dist, prior_latent_sample
    

    def save_rssm(self):
        self.r_model.save()
        self.d_model.save()
        self.e_model.save()
        self.decoder.save()

    def load_rssm(self, custom_path=None):
        if custom_path:
            self.r_model.load(custom_path=custom_path)
            self.e_model.load(custom_path=custom_path)
            self.d_model.load(custom_path=custom_path)
            self.decoder.load(custom_path=custom_path)
        else:
            self.r_model.load()
            self.e_model.load()
            self.d_model.load()
            self.decoder.load()
    







class RSSMTrainer:
    def __init__(self, config, model:RSSM, device) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.optimizer = self.model.optimizer
        self.warmup_epochs = 100
        self.max_kl_weight = float(self.config.kl_weight)
        self.model.to(self.device)

    def get_kl_weight(self, epoch):
        # Linear warmup
        return min(self.max_kl_weight * (epoch / self.warmup_epochs), self.max_kl_weight)
    
    def train(self, data_loader):
        best_loss = float('inf')
        for i in range(self.config.epochs):
            total_recon_loss = 0.0
            total_dynamic_loss = 0.0
            total_rep_loss = 0.0
            total_loss = 0.0
            loop_count = 0

            kl_weight = self.get_kl_weight(i)

            for loop_count, (obs, _, actions, _, _) in enumerate(data_loader, start=1): 

                obs, actions = obs.to(self.device), actions.to(self.device)


                hidden_state, _ = self.model.recurrent_model_input_init()
                
                assert torch.isfinite(obs).all(), "obs contains NaN or Inf"
                assert torch.isfinite(hidden_state).all(), "hidden_state contains NaN or Inf"

                latent_sample, posterior_dist, reconstruct_ob, h, prior_dist, prior_latent_sample = self.model(obs, hidden_state, actions)
                #hidden_state = outputs['h']

                # Calculate Posterior (using next_obs with Encoder)
                #z_posterior, posterior_dist = self.model.e_model(torch.cat((next_obs, hidden_state), dim=-1))

                # Calculate losses
                # Reconstruction Loss for Decoder
                
                recon_loss = F.mse_loss(reconstruct_ob, obs)

                # KL Divergence Loss
                #posterior_dist = outputs['posterior_dist']
                #prior_dist = outputs['prior_dist']

                posterior_dist_stopped = torch.distributions.Normal(
                posterior_dist.loc.detach(),  # Detach the mean
                posterior_dist.scale.detach()  # Detach the standard deviation
                )

                dynamic_loss = torch.distributions.kl_divergence(posterior_dist_stopped, prior_dist).sum(dim=-1).mean()
                dynamic_loss = torch.clamp(dynamic_loss, min=float(self.config.free_bits_threshold))

                prior_dist_stopped = torch.distributions.Normal(
                    prior_dist.loc.detach(),  # Detach the mean
                    prior_dist.scale.detach()  # Detach the standard deviation
                )

                

                rep_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist_stopped).sum(dim=-1).mean()
                rep_loss = torch.clamp(rep_loss, min=float(self.config.free_bits_threshold))
                
                #print(f"reconstruct: {recon_loss}")
                #print(f"reward : {reward_loss}")
                #print(f"Continue : {continue_loss}")
                #print(f"kl weight : {self.config.kl_weight}")
                # Total Loss
                batch_total_loss = recon_loss + kl_weight * dynamic_loss + kl_weight * rep_loss
                
                total_recon_loss += recon_loss.item()
                total_dynamic_loss += dynamic_loss.item()
                total_rep_loss += rep_loss.item()
                total_loss += batch_total_loss.item()  # Add the batch total, not total_loss

                self.optimizer.zero_grad()
                batch_total_loss.backward()
                self.optimizer.step()

            # Calculate average losses for the epoch
            avg_recon_loss = total_recon_loss / loop_count
            avg_dynamic_loss = total_dynamic_loss / loop_count
            avg_rep_loss = total_rep_loss / loop_count
            avg_total_loss = total_loss / loop_count


            print(
            f"Epoch {i + 1}/{self.config.epochs}: "
            f"Avg Total Loss = {avg_total_loss:.4f}, "
            f"Avg Recon Loss = {avg_recon_loss:.4f}, "
            f"Avg Dynamic KL Loss = {avg_dynamic_loss:.4f}, "
            f"Avg Representation Loss = {avg_rep_loss:.4f}"
        )


            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.model.save_rssm()
                print(f"New best model saved with Avg Total Loss = {avg_total_loss:.4f}")   
    