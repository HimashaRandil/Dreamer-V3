import numpy as np
import torch
from collections import deque
from typing import Dict, List, Optional
from ..modules.worldModel import WorldModel
from dreamer.modules.actor_critic import ActorCritic

class ReplayBuffer:
    def __init__(self, capacity: int, actor_critic: ActorCritic, sequence_length: int, device: str = "cuda"):
        """
        Initialize Replay Buffer for DreamerV3.
        
        Args:
            capacity: Maximum number of trajectories to store
            actor_critic: ActorCritic instance for action processing
            sequence_length: Length of each trajectory sequence
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.actor_critic = actor_critic
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def add(self, trajectory: Dict[str, torch.Tensor]) -> None:
        """
        Add a trajectory to the buffer.
        
        Args:
            trajectory: Dictionary containing:
                - 'states': tensor of shape [sequence_length, state_dim] 
                - 'actions': tensor of shape [sequence_length, action_dim]
                - 'rewards': tensor of shape [sequence_length]
                - 'continues': tensor of shape [sequence_length] (1 - done)
        """

        required_keys = ['states', 'actions', 'rewards', 'continues']
        if not all(key in trajectory for key in required_keys):
            raise ValueError(f"Trajectory must contain keys: {required_keys}")
        

        # Verify all items are tensors
        if not all(isinstance(tensor, torch.Tensor) for tensor in trajectory.values()):
            raise ValueError("All trajectory values must be torch.Tensor")

        # Ensure all tensors are on CPU for storage
        # Detach and move to CPU for storage
        trajectory_cpu = {}

        try:
            for key, tensor in trajectory.items():
                # Ensure tensor is detached and moved to CPU
                if tensor.requires_grad:
                    tensor = tensor.detach()
                trajectory_cpu[key] = tensor.cpu()
                
                # Verify tensor isn't sharing memory with original
                if tensor.storage().data_ptr() == trajectory[key].storage().data_ptr():
                    raise RuntimeError(f"Tensor for '{key}' wasn't properly detached")
        except Exception as e:
            raise RuntimeError(f"Failed to process trajectory for storage: {e}")
        self.buffer.append(trajectory_cpu)


        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of trajectories.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            Dictionary containing batched trajectories on specified device
        """
        if len(self.buffer) < batch_size:
            raise RuntimeError(f"Not enough trajectories in buffer. Have {len(self.buffer)}, requested {batch_size}")
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=True)
        
        # Get trajectories
        trajectories = [self.buffer[idx] for idx in indices]
        
        # Explicit device handling
        batch = {}
        for key in trajectories[0].keys():
            try:
                stacked = torch.stack([t[key] for t in trajectories])
                batch[key] = stacked.to(self.device)
            except Exception as e:
                raise RuntimeError(f"Error processing key {key}: {e}")
        
        return batch
    
    def generate_sequence(self, world_model: WorldModel, env, sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate a new trajectory using the world model.
        
        Args:
            world_model: Trained world model
            sequence_length: Optional override for sequence length
            
        Returns:
            Dictionary containing the generated trajectory
        """
        seq_len = sequence_length or self.sequence_length
        
        # Initialize lists to store trajectory components
        states = []  
        actions = []
        rewards = []
        continues = []
        
        try: 
            # Get initial observation from Grid2Op
            obs = env.reset()  # This gives us a Grid2Op observation
            if obs is None:
                raise RuntimeError("Environment reset failed")
            # Convert to tensor and process it according to your observation processing logic
            obs_tensor = torch.from_numpy(obs.to_vect()).float().unsqueeze(0).to(self.device)
            batch_size = obs_tensor.size(0)
            
            with torch.no_grad():
                try:
                    # Encode initial observation using world model's encoder
                    current_z, _ = world_model.rssm.e_model(obs_tensor)
                    current_h, _ = world_model.rssm.recurrent_model_input_init(batch=batch_size)
                    
                    # Store initial combined state
                    current_state = torch.cat([current_z, current_h], dim=-1).detach()
                    states.append(current_state)

                    # Generate initial action for first state
                    action, _ = self.actor_critic.actor.act(current_state)
                    actions.append(action.detach())
                    
                    for t in range(seq_len-1):

                        # Use world model to predict next states
                        action_onehot = self.actor_critic.one_hot_encode(action)
                        next_h = world_model.rssm.r_model(current_z, action_onehot, current_h)
                        _, next_z = world_model.rssm.d_model(next_h)
                        
                        reward = world_model.reward_predictor(next_z, next_h)
                        cont = world_model.continue_predictor(next_z, next_h)
                        
                        # Store states and predictions
                        next_state = torch.cat([next_z, next_h], dim=-1).detach()
                        states.append(next_state)
                        rewards.append(reward.detach())
                        continues.append(cont.detach())

                        # Get action for next state
                        action, _ = self.actor_critic.actor.act(next_state)
                        actions.append(action.detach())
                                    
                        current_z = next_z.detach()
                        current_h = next_h.detach()
                        current_state = next_state

                except Exception as model_error:
                    raise RuntimeError(f"World model prediction failed: {model_error}")
                    
            
            try:
                # Stack all tensors
                trajectory = {
                    'states': torch.stack(states),        # [seq_len, batch_size, state_dim]
                    'actions': torch.stack(actions),      # [seq_len, batch_size, action_dim]
                    'rewards': torch.stack(rewards),      # [seq_len, batch_size]
                    'continues': torch.stack(continues)   # [seq_len, batch_size]
                }

                # Verify trajectory shapes
                expected_length = seq_len
                for key, tensor in trajectory.items():
                    if tensor.size(0) != expected_length:
                        raise ValueError(
                            f"Trajectory tensor '{key}' has incorrect length. "
                            f"Expected {expected_length}, got {tensor.size(0)}"
                        )
            
                return trajectory
            
            except Exception as e:
                raise RuntimeError(f"Failed to create trajectory: {e}")
            
        except Exception as e:
            raise RuntimeError(f"Sequence generation failed: {e}")   
    

    def initialize_buffer(self, world_model: WorldModel, env, num_trajectories: int) -> None:
        """
        Initialize the buffer with trajectories generated from the world model.
        
        Args:
            world_model: Trained world model
            env: Grid2Op environment instance
            num_trajectories: Number of trajectories to generate
        """
        print(f"Generating {num_trajectories} initial trajectories...")
        
        for i in range(num_trajectories):
            if i % 10 == 0:
                print(f"Generated {i}/{num_trajectories} trajectories")
                
            trajectory = self.generate_sequence(world_model,env)
            self.add(trajectory)
            
        print("Buffer initialization complete!")
    
    def __len__(self) -> int:
        return len(self.buffer)