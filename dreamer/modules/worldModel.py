import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from dreamer.modules.rssm import RSSM
from dreamer.modules.decoder import Decoder
from dreamer.modules.rewardModel import RewardPredictor
from dreamer.modules.continueModel import ContinuousPredictor
import torch.optim as optim

class WorldModel(nn.Module):
    def __init__(self, config) -> None:
        super(WorldModel, self).__init__()

        self.config = config
        self.rssm = RSSM(config)
        self.decoder = Decoder(config)
        self.reward_predictor = RewardPredictor(config)
        self.continue_predictor = ContinuousPredictor(config)



    def forward(self, obs, hidden_state, action):
        z_prior, z_posterior, h, dist, dynamic_dist = self.rssm(obs, hidden_state, action)   #Dynamics Predictor: Provides the prior distribution

        reconstructed_obs = self.decoder(z_posterior, h)
        reward = self.reward_predictor(z_posterior, h)
        continue_prob = self.continue_predictor(z_posterior, h)

        return {
            'prior_dist': dynamic_dist,
            'posterior_dist': dist,
            'reconstructed_obs': reconstructed_obs,
            'reward': reward,
            'continue_prob': continue_prob,
            'hidden_state': h
        }
    


class Trainer:
    def __init__(self, config, model:WorldModel) -> None:
        self.config = config
        self.model = model

        self.model.to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    
    def train(self, data_loader):
        for obs, actions, rewards, dones, next_obs in data_loader:  # Assume batched data from replay buffer
            obs, actions, rewards, dones, next_obs = obs.to(self.config.device), actions.to(self.config.device), rewards.to(self.config.device), dones.to(self.config.device), next_obs.to(self.config.device)
        
            hidden_state, action = self.model.rssm.recurrent_model_input_init()

            outputs = self.model(obs, hidden_state, action)

            # Calculate Posterior (using next_obs with Encoder)
            z_posterior, posterior_dist = self.model.e_model(torch.cat((next_obs, hidden_state), dim=-1))

            # Calculate losses
            # Reconstruction Loss for Decoder
            reconstructed_obs = outputs['reconstructed_obs']
            recon_loss = F.mse_loss(reconstructed_obs, obs)

            # Reward Predictor Loss
            reward_pred = outputs['reward']
            reward_loss = F.mse_loss(reward_pred, rewards)

            # Continue Predictor Loss (Binary Cross-Entropy)
            continue_pred = outputs['continue_prob']
            continue_loss = F.binary_cross_entropy_with_logits(continue_pred, dones.float())

            # KL Divergence Loss
            posterior_dist = outputs['posterior_dist']
            prior_dist = outputs['prior_dist']
            kl_div = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
            
            # Apply free bits by clipping the KL divergence
            kl_div = torch.clamp(kl_div, min=self.config.free_bits_threshold)

            # Total Loss
            total_loss = recon_loss + reward_loss + continue_loss + self.config.kl_weight * kl_div
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


            print(f"Total Loss: {total_loss.item()}, Recon Loss: {recon_loss.item()}, Reward Loss: {reward_loss.item()}, Continue Loss: {continue_loss.item()}, KL Loss: {kl_div.item()}")

    
    def evaluate(self, data_loader):
        pass

    def evaluate_with_grid(self):
        pass

    def save_world_model(self):
        pass

    def load_world_model(self):
        pass

