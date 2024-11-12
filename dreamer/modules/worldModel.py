import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from dreamer.modules.rssm import RSSM
from dreamer.modules.decoder import Decoder
from dreamer.modules.rewardModel import RewardPredictor
from dreamer.modules.continueModel import ContinuousPredictor


class WorldModel(nn.Module):
    def __init__(self, config) -> None:
        super(WorldModel, self).__init__()

        self.config = config
        self.rssm = RSSM(config)
        self.decoder = Decoder(config)
        self.reward_predictor = RewardPredictor(config)
        self.continue_predictor = ContinuousPredictor(config)



    def forward(self, obs, hidden_state, action):
        z_prior, z_posterior, h, dist = self.rssm(obs, hidden_state, action)   #Dynamics Predictor: Provides the prior distribution

        reconstructed_obs = self.decoder(z_posterior, h)
        reward = self.reward_predictor(z_posterior, h)
        continue_prob = self.continue_predictor(z_posterior, h)

        return {
            'prior_dist': z_prior,
            'posterior_dist': dist,
            'reconstructed_obs': reconstructed_obs,
            'reward': reward,
            'continue_prob': continue_prob,
            'hidden_state': h
        }


