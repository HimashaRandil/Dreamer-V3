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
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
        
        self.r_model = RecurrentModel(self.config)
        self.d_model = DynamicPredictor(self.config)
        self.e_model = Encoder(self.config)


    def recurrent_model_input_init(self):
        return self.r_model.input_init(), self.r_model.action_init()

    
    def forward(self, obs, hidden_state, action):
        x = torch.cat((obs, hidden_state), dim=-1)
        z, dist = self.e_model(x)
        h = self.r_model(z, action, hidden_state)
        z_t, dynamic_dist = self.d_model(h)
        return z_t, z, h, dist, dynamic_dist
    

