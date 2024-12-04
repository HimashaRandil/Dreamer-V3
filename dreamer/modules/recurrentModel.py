import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class RecurrentModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(RecurrentModel, self).__init__()
        
        self.latent_dim = config.latent_dim      # Dimension of the latent state z_t
        self.action_dim = config.action_dim      # Dimension of the action space a_t
        self.hidden_dim = config.hidden_dim      # Dimension of the hidden state h_t in the RNN
        self.config = config
        
        # Define the GRU layer
        self.gru = nn.GRUCell(input_size=self.latent_dim + self.action_dim, 
                              hidden_size=self.hidden_dim)
        
        # Initialize a linear layer to project the concatenated input into the GRU input
        self.input_proj = nn.Linear(self.latent_dim + self.action_dim, self.latent_dim + self.action_dim)

        if kwargs.get('path'):
            self.config.path = kwargs.get('path')

    def forward(self, z_t, a_t, h_t_prev):
        combined_input = torch.cat([z_t, a_t], dim=-1)
        
        gru_input = self.input_proj(combined_input)
    
        h_t = self.gru(gru_input, h_t_prev)
        
        return h_t 
    

    def input_init(self, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        return torch.zeros(batch_size, self.config.hidden_dim).to(self.config.device)
    
    def action_init(self):
        return torch.zeros(self.config.batch_size, self.config.action_dim).to(self.config.device)
    

    def save(self, model_name="recurrent_model"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="recurrent_model", custom_path=None):
        if custom_path:
            model_path = os.path.join(custom_path, model_name)
        else:
            model_path = os.path.join(self.config.saved_model_path, model_name)
    
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Model file not found at {model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")