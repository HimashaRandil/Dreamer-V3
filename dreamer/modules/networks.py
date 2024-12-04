import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from typing import List 


TensorTBCHW = Tensor
TensorTB = Tensor
TensorTBE = Tensor
TensorTBICHW = Tensor
TensorTBIF = Tensor
TensorTBI = Tensor
TensorJMF = Tensor
TensorJM2 = Tensor
TensorHMA = Tensor
TensorHM = Tensor
TensorJM = Tensor

IntTensorTBHW = Tensor
# StateB = Tuple[Tensor, Tensor]
# StateTB = Tuple[Tensor, Tensor]




# Utility functions for batch flattening and unflattening
def flatten_batch(x: Tensor) -> Tensor:
    batch_size = x.size(0)
    flattened = x.view(-1, x.size(-1))  # Flatten all except the last dimension
    return flattened, batch_size

def unflatten_batch(x: Tensor, batch_size: int) -> Tensor:
    return x.view(batch_size, -1)




class NoNorm(nn.Module):

    """Simple pass-through normalization layer."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x




class ActorNetwork(nn.Module):

    '''
    Actor Network for discrete action spaces
    
    Features:
    - Configurable architecture with optional layer normalization
    - Support for both training and inference modes
    - Built-in action sampling and evaluation
    - Proper handling of batched inputs
    
    '''
    def __init__(self, in_dim: int, action_dim: int, actor_hidden_dims: List[int], layer_norm: bool = True,
        activation: nn.Module = nn.ELU, epsilon: float = 1e-3):

        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        norm = nn.LayerNorm if layer_norm else NoNorm

        layers = []
        current_dim = in_dim

        for hidden_dim in actor_hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                norm(hidden_dim, eps=epsilon),
                activation()
            ])
            current_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.model = nn.Sequential(*layers)



    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Actor Network.
        Args:
            x: Input tensor (batch_size, in_dim) or (T, B, in_dim)
        Returns:
            logits: Action logits of shape (batch_size, action_dim) or (T, B, action_dim)

        """

        assert x.dim() in [2, 3], f"Input must be 2D or 3D tensor, got shape {x.shape}"
        assert x.size(-1) == self.in_dim, f"Expected input feature size {self.in_dim}, got {x.size(-1)}"
        

        original_shape = x.shape[:-1]  # Store original batch dimensions
        x = x.view(-1, x.size(-1))     # Flatten batch dimensions
        logits = self.model(x)         # (N, action_dim)

        # Verify output dimensions
        assert logits.size(-1) == self.action_dim, f"Network output dimension mismatch. Expected {self.action_dim}, got {logits.size(-1)}"

        # Restore original batch dimensions
        return logits.view(*original_shape, self.action_dim)
    

    def sample_action(self, logits: Tensor, deterministic: bool = False, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            logits: Action logits from the forward pass
            deterministic: If True, return the most likely action
            temperature: Temperature for sampling (higher = more random)
        
        Returns:
            Tuple of:
                - actions: Sampled actions
                - log_probs: Log probabilities of sampled actions
        """

        assert temperature > 0, f"Temperature must be positive, got {temperature}"

        try: 

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                actions = torch.argmax(probs, dim=-1)
                log_probs = torch.log(torch.gather(probs, -1, actions.unsqueeze(-1))).squeeze(-1)
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                
            return actions, log_probs
        
        except Exception as e:
            raise RuntimeError(f"Error sampling action: {str(e)}")


    def evaluate_actions(self, logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            logits: Action logits from the forward pass
            actions: Actions to evaluate
            
        Returns:
            Tuple of:
                - log_probs: Log probabilities of the actions
                - entropy: Entropy of the policy distribution
                - probs: Action probabilities
        """
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, probs
    
    @torch.no_grad()
    def act(
        self,
        state: Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Convenience method for getting actions from states.
        
        Args:
            state: Input state tensor
            deterministic: Whether to sample deterministically
            temperature: Sampling temperature
            
        Returns:
            Tuple of:
                - actions: Sampled actions
                - log_probs: Log probabilities of sampled actions
        """
        logits = self.forward(state)
        return self.sample_action(logits, deterministic, temperature)
    
    def get_model_summary(self) -> None:
        """Print a summary of the network architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\nActor Network Architecture:")
        print("-" * 40)
        
        current_dim = None
        for idx, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                current_dim = layer.out_features
                print(f"Layer {idx}: Linear ({layer.in_features} → {layer.out_features})")
            elif isinstance(layer, nn.LayerNorm):
                print(f"Layer {idx}: LayerNorm ({current_dim})")
            else:
                print(f"Layer {idx}: {layer.__class__.__name__}")
                
        print("-" * 40)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("-" * 40)




class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=400, num_buckets: int = 255):
        """
        Critic Network
        
        Args:
            obs_dim (int): Dimension of the observation space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Dimension of hidden layers
            num_buckets: Number of buckets for value discretization

        """

        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_buckets = num_buckets

        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )


        # Action processing layers
        self.action_net = nn.Sequential(
            nn.Linear(action_dim,hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ELU(),
        )

        # Combined processing Layer
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim+hidden_dim//2 , hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # Value prediction layers with dual heads (similar to Dreamer v3)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_buckets, bias=False)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Initialize output layers to zero
        self._initialize_zero_weights()
        
    def _initialize_zero_weights(self):
        """Initialize output layer weights to zero as per Dreamer v3."""
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.risk_head[-1].weight)


    def _process_action(self, action: torch.Tensor) -> torch.Tensor:
        """Process action input to correct format."""
        if action.dim() == 1:
            # Convert discrete action to one-hot
            return F.one_hot(action, num_classes=self.action_dim).float()
        return action


    def forward(self, obs, action):
        """
        Forward pass of the critic network.
        
        Args:
            obs (torch.Tensor): Observation tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            Tuple of:
                - value: Expected value computed from bucket distribution
                - risk: Risk prediction
        """

        # Process action to one-hot if needed
        # action = self._process_action(action)
            
        # Extract features from observation

        assert obs.dim() in [2,3], f"Observation must be 2D or 3D tensor, got shape {obs.shape}"
        assert action.dim() in [2,3], f"Action must be 2D or 3D tensor, got shape {action.shape}"
        assert obs.size(-1) == self.obs_dim, f"Expected observation feature size {self.obs_dim}, got {obs.size(-1)}"
        assert action.size(-1) == self.action_dim, f"Expected action feature size {self.action_dim}, got {action.size(-1)}"


        features = self.feature_net(obs)
        action_features = self.action_net(action) # Process action
        combined = torch.cat([features, action_features], dim=-1)# Combine features
        combined_features = self.combined_net(combined)# Combine features
        value_logits = self.value_head(combined_features) # Predict value and risk
        risk = self.risk_head(combined_features)# Predict value and risk

        # Convert value distribution to scalar value
        value_probs = F.softmax(value_logits, dim=-1)
        bucket_values = torch.linspace(-20, 20, self.num_buckets, device=obs.device)
        value = (value_probs * bucket_values).sum(dim=-1, keepdim=True)
        
        return value, risk
    
    def get_value_distribution(self, obs, action):
        assert obs.dim() in [2,3], f"Observation must be 2D or 3D tensor, got shape {obs.shape}"
        assert action.dim() in [2,3], f"Action must be 2D or 3D tensor, got shape {action.shape}"
        assert obs.size(-1) == self.obs_dim, f"Expected observation feature size {self.obs_dim}, got {obs.size(-1)}"
        assert action.size(-1) == self.action_dim, f"Expected action feature size {self.action_dim}, got {action.size(-1)}"


        features = self.feature_net(obs)
        action_features = self.action_net(action) # Process action
        combined = torch.cat([features, action_features], dim=-1)# Combine features
        combined_features = self.combined_net(combined)# Combine features
        value_logits = self.value_head(combined_features)

        return value_logits 

    

    def predict_value(self, obs, action):
        """
        Predict only the value for a given observation-action pair.
        
        Args:
            obs (torch.Tensor): Observation tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            torch.Tensor: Value prediction
        """
        value, _ = self.forward(obs, action)
        return value
    
    def predict_risk(self, obs, action):
        """
        Predict only the risk for a given observation-action pair.
        
        Args:
            obs (torch.Tensor): Observation tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            torch.Tensor: Risk prediction
        """
        _, risk = self.forward(obs, action)
        return risk







# ----------------------------------------------------------------------------------------------------------------------- # 


#  Following is an additional Actor network thats under construction 
class DreamerV3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim, recurrent=False, activation=nn.Tanh):
        super(DreamerV3Actor, self).__init__()
        
        self.recurrent = recurrent
        self.latent_dim = latent_dim
        
        # Define a basic MLP backbone with latent dimension for DreamerV3
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Latent space for DreamerV3 (can be used with VAE)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

        if self.recurrent:
            # Recurrent part if required (GRU or LSTM)
            self.rnn = nn.GRU(latent_dim, latent_dim)
        
        # Action distribution output
        self.fc_action = nn.Linear(latent_dim, action_dim)
    
    def forward(self, state):
        # Process state through MLP
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Learn latent representation (can also use VAE)
        latent = self.fc_latent(x)
        
        if self.recurrent:
            # Handle recurrent processing
            latent, _ = self.rnn(latent.unsqueeze(0))  # Add batch dimension for RNN
            
        # Output action probabilities (logits for discrete or mean/variance for continuous)
        logits = self.fc_action(latent)
        
        return logits
    
    def sample_action(self, logits):
        # Assuming logits are from a categorical distribution for discrete actions
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action


        