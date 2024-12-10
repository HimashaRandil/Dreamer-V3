import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from dreamer.modules.rssm import RSSM
from dreamer.modules.decoder import Decoder
from dreamer.modules.rewardModel import RewardPredictor
from dreamer.modules.continueModel import ContinuousPredictor
import torch.optim as optim
import grid2op
from grid2op.Reward import L2RPNSandBoxScore
from lightsim2grid import LightSimBackend
from typing import Tuple
from dreamer.modules.networks import ActorNetwork



class WorldModel(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super(WorldModel, self).__init__()
        self.config = config
        if kwargs.get('path'):
            path = kwargs.get('path')

            self.rssm = RSSM(config, path=path)
            self.decoder = Decoder(config, path=path)
            self.reward_predictor = RewardPredictor(config, path=path)
            self.continue_predictor = ContinuousPredictor(config, path=path)

        self.rssm = RSSM(config)
        self.decoder = Decoder(config)
        self.reward_predictor = RewardPredictor(config)
        self.continue_predictor = ContinuousPredictor(config)

        if torch.cuda.is_available():
            print("Cuda is availabel for Training")
        else:
            print("Cuda is not available for training")
            
        self.device = torch.device(self.config.device)

        self.optimizer = optim.Adam([
                {'params': self.rssm.r_model.parameters()},  
                {'params': self.rssm.d_model.parameters()},
                {'params': self.rssm.e_model.parameters()},
                {'params': self.decoder.parameters()},  
                {'params': self.reward_predictor.parameters()},
                {'params': self.continue_predictor.parameters()}
            ], lr=float(self.config.learning_rate))
        

        self.to(self.device)



    def forward(self, obs, hidden_state, action):
        z_prior, z_posterior, h, dist, dynamic_dist = self.rssm(obs, hidden_state, action)   #Dynamics Predictor: Provides the prior distribution

        reconstructed_obs = self.decoder(z_posterior, hidden_state)
        reward_dist = self.reward_predictor(z_posterior, hidden_state)
        continue_prob = self.continue_predictor(z_posterior, hidden_state)

        return {
            'prior_dist': dynamic_dist,
            'posterior_dist': dist,
            'reconstructed_obs': reconstructed_obs,
            'reward_dist': reward_dist,
            'continue_prob': continue_prob,
            'hidden_state': h
        }
    

    def save_world_model(self):
        self.rssm.save_rssm()
        self.reward_predictor.save()
        self.continue_predictor.save()
        self.decoder.save()
        print(f"World Model saved at {self.config.path}")

    def load_world_model(self, custom_path=None):
        if custom_path:
            self.rssm.load_rssm(custom_path=custom_path)
            self.reward_predictor.load(custom_path=custom_path)
            self.continue_predictor.load(custom_path=custom_path)
            self.decoder.load(custom_path=custom_path)
            print(f"World model loaded at {custom_path}")
        else:
            self.rssm.load_rssm()
            self.reward_predictor.load()
            self.continue_predictor.load()
            self.decoder.load()
            print(f"World model loaded at {self.config.saved_model_path}")


    
    


class Trainer:
    def __init__(self, config, model:WorldModel) -> None:
        self.config = config
        self.model = model

        self.optimizer = self.model.optimizer
        self.warmup_epochs = 100
        self.max_kl_weight = float(self.config.kl_weight)

    def get_kl_weight(self, epoch):
        # Linear warmup
        return min(self.max_kl_weight * (epoch / self.warmup_epochs), self.max_kl_weight)
    
    def train(self, data_loader):
        best_loss = float('inf')
        for i in range(self.config.epochs):
            total_recon_loss = 0.0
            total_reward_loss = 0.0
            total_continue_loss = 0.0
            total_dynamic_loss = 0.0
            total_rep_loss = 0.0
            total_loss = 0.0
            loop_count = 0

            kl_weight = self.get_kl_weight(i)

            for loop_count, (obs, rewards, actions, dones, next_obs) in enumerate(data_loader, start=1): 

                obs, rewards, actions, dones, next_obs = obs.to(self.model.device), rewards.to(self.model.device), actions.to(self.model.device), dones.unsqueeze(1).to(self.model.device), next_obs.to(self.model.device)

                num_true = dones.sum().item()
                num_false = dones.numel() - num_true
                total = num_true + num_false
                weight_true = total / (2 * num_true) if num_true > 0 else 1.0
                weight_false = total / (2 * num_false) if num_false > 0 else 1.0

                # Create weights tensor for the batch
                batch_weights = dones.float() * weight_true + (1 - dones.float()) * weight_false

                hidden_state, _ = self.model.rssm.recurrent_model_input_init()
                
                assert torch.isfinite(obs).all(), "obs contains NaN or Inf"
                assert torch.isfinite(hidden_state).all(), "hidden_state contains NaN or Inf"

                outputs = self.model(obs, hidden_state, actions)
                #hidden_state = outputs['h']

                # Calculate Posterior (using next_obs with Encoder)
                #z_posterior, posterior_dist = self.model.e_model(torch.cat((next_obs, hidden_state), dim=-1))

                # Calculate losses
                # Reconstruction Loss for Decoder
                reconstructed_obs = outputs['reconstructed_obs']
                recon_loss = F.mse_loss(reconstructed_obs, obs)

                # Reward Predictor Loss
                reward_pred = outputs['reward_dist']
                std_target = 0.1  # Fixed standard deviation for target rewards
                target_dist = torch.distributions.Normal(rewards, std_target)
                reward_loss = torch.distributions.kl_divergence(reward_pred, target_dist).sum(dim=-1).mean()
                reward_loss = torch.clamp(reward_loss, min=float(self.config.free_bits_threshold))

                #reward_loss = F.mse_loss(reward_pred, rewards)

                # Continue Predictor Loss (Binary Cross-Entropy)
                continue_pred = outputs['continue_prob']
                continue_loss = F.binary_cross_entropy_with_logits(
                            continue_pred, dones, weight=batch_weights)

                # KL Divergence Loss
                posterior_dist = outputs['posterior_dist']
                prior_dist = outputs['prior_dist']

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
                batch_total_loss = recon_loss + reward_loss + continue_loss + kl_weight * dynamic_loss + kl_weight * rep_loss
                
                total_recon_loss += recon_loss.item()
                total_reward_loss += reward_loss.item()
                total_continue_loss += continue_loss.item()
                total_dynamic_loss += dynamic_loss.item()
                total_rep_loss += rep_loss.item()
                total_loss += batch_total_loss.item()  # Add the batch total, not total_loss

                self.optimizer.zero_grad()
                batch_total_loss.backward()
                self.optimizer.step()

            # Calculate average losses for the epoch
            avg_recon_loss = total_recon_loss / loop_count
            avg_reward_loss = total_reward_loss / loop_count
            avg_continue_loss = total_continue_loss / loop_count
            avg_dynamic_loss = total_dynamic_loss / loop_count
            avg_rep_loss = total_rep_loss / loop_count
            avg_total_loss = total_loss / loop_count


            print(
            f"Epoch {i + 1}/{self.config.epochs}: "
            f"Avg Total Loss = {avg_total_loss:.4f}, "
            f"Avg Recon Loss = {avg_recon_loss:.4f}, "
            f"Avg Reward Loss = {avg_reward_loss:.4f}, "
            f"Avg Continue Loss = {avg_continue_loss:.4f}, "
            f"Avg Dynamic KL Loss = {avg_dynamic_loss:.4f}, "
            f"Avg Representation Loss = {avg_rep_loss:.4f}"
        )


            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.model.save_world_model()
                print(f"New best model saved with Avg Total Loss = {avg_total_loss:.4f}")   
            
    
    def evaluate(self, data_loader):
        self.model.eval()

        total_recon_loss = 0
        total_reward_loss = 0
        total_continue_loss = 0
        total_kl_div = 0
        num_batches = 0

        with torch.no_grad():  # Disable gradient computation
            for obs, actions, rewards, dones, next_obs in data_loader:
                # Move data to the appropriate device
                obs, actions, rewards, dones = obs.to(self.config.device), actions.to(self.config.device), rewards.to(self.config.device), dones.to(self.config.device)
                next_obs = next_obs.to(self.config.device)

                # Initialize hidden state and action at t=0
                hidden_state, action = self.model.rssm.recurrent_model_input_init()

                # Forward pass through the world model
                outputs = self.model(obs, hidden_state, action)
                hidden_state = outputs['h']

                # Calculate Posterior (using next_obs with Encoder)
                z_posterior, posterior_dist = self.model.e_model(torch.cat((next_obs, hidden_state), dim=-1))

                # Reconstruction Loss (Decoder)
                reconstructed_obs = outputs['reconstructed_obs']
                recon_loss = F.mse_loss(reconstructed_obs, obs, reduction='sum')
                total_recon_loss += recon_loss.item()

                # Reward Prediction Loss (MSE)
                reward_pred = outputs['reward']
                reward_loss = F.mse_loss(reward_pred, rewards, reduction='sum')
                total_reward_loss += reward_loss.item()

                # Continue Predictor Loss (Binary Cross-Entropy)
                continue_pred = outputs['continue_prob']
                continue_loss = F.binary_cross_entropy_with_logits(continue_pred, dones.float(), reduction='sum')
                total_continue_loss += continue_loss.item()

                # KL Divergence Loss
                #posterior_dist = outputs['posterior_dist']
                prior_dist = outputs['prior_dist']
                kl_div = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
                total_kl_div += kl_div.item()

                # Increment the batch counter
                num_batches += 1

        # Calculate average metrics
        avg_recon_loss = total_recon_loss / len(data_loader.dataset)
        avg_reward_loss = total_reward_loss / len(data_loader.dataset)
        avg_continue_loss = total_continue_loss / len(data_loader.dataset)
        avg_kl_div = total_kl_div / num_batches

        # Print or log metrics
        print(f"Evaluation - Recon Loss: {avg_recon_loss:.4f}, Reward Loss: {avg_reward_loss:.4f}, Continue Loss: {avg_continue_loss:.4f}, KL Divergence: {avg_kl_div:.4f}")

        # Return metrics as a dictionary
        return {
            'recon_loss': avg_recon_loss,
            'reward_loss': avg_reward_loss,
            'continue_loss': avg_continue_loss,
            'kl_divergence': avg_kl_div
        }
    

    def evaluate_with_grid(self):
        env = grid2op.make(self.config.env_name, reward_class=L2RPNSandBoxScore,
                                backend=LightSimBackend())
        
    
    
        
        

    

