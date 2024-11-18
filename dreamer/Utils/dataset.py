import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

def load_npz_files_from_folder(folder_path):
    all_observations = []
    all_rewards = []
    all_actions = []
    all_dones = []
    all_next_observations = []
    
    # Iterate through all .npz files in the folder
    for filename in os.listdir(folder_path):
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
    
    return observations, rewards, actions, dones, next_observations



class GrdiDataset(Dataset):
    def __init__(self, observations, rewards, actions, dones, next_observations):
        self.observations = torch.tensor(np.array(observations, np.float32), dtype=torch.float32)
        self.rewards = torch.tensor(np.array(rewards, np.float32), dtype=torch.float32)
        self.actions = torch.tensor(np.array(actions, np.float32), dtype=torch.long)  # Assuming discrete actions
        self.dones = torch.tensor(np.array(dones, np.float32), dtype=torch.float32)  # Done flags as float
        self.next_observations = torch.tensor(np.array(next_observations, np.float32), dtype=torch.float32)
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return (self.observations[idx], self.rewards[idx], self.actions[idx], self.dones[idx], self.next_observations[idx])
