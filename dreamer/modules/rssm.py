import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.r_model = RecurrentModel(self.config)
        self.d_model = DynamicPredictor(self.config)
        self.e_model = Encoder(self.config)

    def recurrent_model_input_init(self):
        return self.r_model.input_init(), self.r_model.action_init()

    
    def forward(self, obs, hidden_state, action):
        # Assert feature dimensions
        assert obs.shape[1] == self.config.input_dim, f"Expected obs feature size {self.config.input_dim}, got {obs.shape[1]}"
        assert hidden_state.shape[1] == self.config.hidden_dim, f"Expected hidden_state size {self.config.hidden_dim}, got {hidden_state.shape[1]}"
        assert action.shape[1] == self.config.action_dim, f"Expected action size {self.config.action_dim}, got {action.shape[1]}"


        x = torch.cat([obs, hidden_state], dim=-1)
        assert x.shape[1] == self.config.input_dim + self.config.hidden_dim

        z, dist = self.e_model(x)
        h = self.r_model(z, action, hidden_state)
        dynamic_dist, z_t = self.d_model(h)
        return z_t, z, h, dist, dynamic_dist
    

    def save_rssm(self):
        self.r_model.save()
        self.d_model.save()
        self.e_model.save()

    def load_rssm(self, custom_path=None):
        if custom_path:
            self.r_model.load(custom_path=custom_path)
            self.e_model.load(custom_path=custom_path)
            self.d_model.load(custom_path=custom_path)
        else:
            self.r_model.load()
            self.e_model.load()
            self.d_model.load()
    

