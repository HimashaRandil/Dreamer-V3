import numpy as np
import torch
from collections import deque
from typing import Dict, List, Optional

class ReplayBuffer:
    def __init__(self, capacity: int, sequence_length: int):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        
    def add(self, trajectory: Dict[str, torch.Tensor]) -> None:
        required_keys = ['states', 'actions', 'rewards', 'continues']
        if not all(key in trajectory for key in required_keys):
            raise ValueError(f"Trajectory must contain keys: {required_keys}")

        # Simplified storage - tensors should already be detached
        self.buffer.append({
            key: tensor.cpu() for key, tensor in trajectory.items()
        })

    def get_sample(self, batch_size: int, device: str = "cuda") -> Dict[str, torch.Tensor]:
        if len(self.buffer) < batch_size:
            raise RuntimeError(f"Not enough trajectories in buffer. Have {len(self.buffer)}, requested {batch_size}")
        
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=True)
        trajectories = [self.buffer[idx] for idx in indices]
        
        return {
            key: torch.stack([t[key] for t in trajectories]).to(device)
            for key in trajectories[0].keys()
        }
    
    
    def __len__(self) -> int:
        return len(self.buffer)
