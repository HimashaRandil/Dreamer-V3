import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dreamer.Utils.utils import Config


config = Config.from_yaml('config.yml')



def one_hot_encode(actions, num_actions):
    """
    Convert an array of actions into one-hot encoding.
    
    Args:
    - actions (np.array): Array of integer actions.
    - num_actions (int): Total number of possible actions (size of action space).
    
    Returns:
    - np.array: One-hot encoded actions, shape (num_samples, num_actions).
    """
    actions = np.array(actions)  
    one_hot_actions = np.zeros((len(actions), num_actions), dtype=np.float32)
    one_hot_actions[np.arange(len(actions)), actions] = 1
    return np.array(one_hot_actions)


def load_npz_files_from_folder(folder_path, start=0, end=100):
    all_observations = []
    all_rewards = []
    all_actions = []
    all_dones = []
    all_next_observations = []
    folder = os.listdir(folder_path)
    # Iterate through all .npz files in the folder
    for filename in folder[start:end]:
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)
            npz_data = np.load(file_path, allow_pickle=True)
            
            all_observations.append(npz_data['obs'])
            all_rewards.append(npz_data['reward'])
            all_actions.append(npz_data['action'])
            all_dones.append(npz_data['done'])
            all_next_observations.append(npz_data['obs_next'])
    

    # Concatenate all arrays along the first axis (stacking the data)
    observations = np.concatenate(all_observations, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    dones = np.concatenate(all_dones, axis=0)
    next_observations = np.concatenate(all_next_observations, axis=0)

    one_hot_actions = one_hot_encode(actions, config.action_dim)
    
    return observations, rewards, one_hot_actions, dones, next_observations



class GrdiDataset(Dataset):
    def __init__(self, observations, rewards, actions, dones, next_observations, device):
        self.observations = torch.tensor(np.array(observations, np.float32), dtype=torch.float32, device=device)
        self.rewards = torch.tensor(np.array(rewards, np.float32), dtype=torch.float32, device=device)
        self.actions = torch.tensor(actions, dtype=torch.int32, device=device)  # Assuming discrete actions
        self.dones = torch.tensor(np.array(dones, np.float32), dtype=torch.float32, device=device)  # Done flags as float
        self.next_observations = torch.tensor(np.array(next_observations, np.float32), dtype=torch.float32, device=device)
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return (self.observations[idx], self.rewards[idx], self.actions[idx], self.dones[idx], self.next_observations[idx])



import os
import numpy as np
import torch

def one_hot_encode(actions, num_actions):
    """
    Convert an array of actions into one-hot encoding.
    
    Args:
    - actions (np.array): Array of integer actions.
    - num_actions (int): Total number of possible actions (size of action space).
    
    Returns:
    - np.array: One-hot encoded actions, shape (num_samples, num_actions).
    """
    actions = np.array(actions)  
    one_hot_actions = np.zeros((len(actions), num_actions), dtype=np.float32)
    one_hot_actions[np.arange(len(actions)), actions] = 1
    return np.array(one_hot_actions)

class LazyGrdiBatchLoader:
    def __init__(self, folder_path, config, device, drop_last=True):
        self.config = config
        self.folder_path = folder_path
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])  # Sort to load files in order
        self.batch_size = self.config.batch_size
        self.device = device
        self.current_data = None
        self.current_index = 0
        self.file_idx = 0
        self.drop_last = drop_last

    def _load_next_file(self):
        if self.file_idx >= len(self.files):
            return None  # No more files to load

        file_path = os.path.join(self.folder_path, self.files[self.file_idx])
        npz_data = np.load(file_path, allow_pickle=True)
        self.file_idx += 1
        self.current_index = 0  # Reset index for new file

        print(f"Loading file: {file_path}")
        return {
            "obs": torch.tensor(np.array(npz_data['obs'], np.float32), dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(np.array(npz_data['reward'], np.float32), dtype=torch.float32, device=self.device),
            "actions": torch.tensor(one_hot_encode(npz_data['action'], self.config.action_dim), dtype=torch.int32, device=self.device),
            "dones": torch.tensor(np.array(npz_data['done'], np.float32), dtype=torch.float32, device=self.device) ,
            "next_obs": torch.tensor(np.array(npz_data['obs_next'], np.float32), dtype=torch.float32, device=self.device),
        }

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # Load new file if current data is exhausted
            if self.current_data is None or self.current_index >= len(self.current_data['obs']):
                self.current_data = self._load_next_file()
                if self.current_data is None:
                    raise StopIteration  # No more data to iterate over

            # Select the batch
            start_idx = self.current_index
            end_idx = min(self.current_index + self.batch_size, len(self.current_data['obs']))
            self.current_index = end_idx

            # Check batch size and skip if smaller than required and drop_last=True
            if self.drop_last and (end_idx - start_idx) < self.batch_size:
                continue

            batch = {
                "obs": self.current_data["obs"][start_idx:end_idx],
                "rewards": self.current_data["rewards"][start_idx:end_idx],
                "actions": self.current_data["actions"][start_idx:end_idx],  # One-hot encoded actions
                "dones": self.current_data["dones"][start_idx:end_idx],
                "next_obs": self.current_data["next_obs"][start_idx:end_idx],
            }
            return batch

"""
folder_path = "dreamer\\data_generation\\temp"
batch_size = 32
num_actions = 10  # Number of possible actions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader = LazyGrdiBatchLoader(folder_path, batch_size, device, num_actions, drop_last=True)

for batch in data_loader:
    print(f"obs shape: {batch['obs'].shape}")"""
