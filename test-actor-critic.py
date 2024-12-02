import torch
import torch.nn as nn
import torch.nn.functional as F  # Added this import
import numpy as np
from typing import NamedTuple
import yaml
from dreamer.modules.networks import ActorNetwork, CriticNetwork


def load_config(yaml_path):
    """Load configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def generate_synthetic_episode(config, batch_size):
    """Generate synthetic episode data"""
    states = torch.randn(config['horizon'], batch_size, config['input_dim'])
    rewards = torch.randn(config['horizon'], batch_size, 1) * 0.1  # Small random rewards
    dones = torch.zeros(config['horizon'], batch_size, 1)  # No episodes end
    return states, rewards, dones




def test_actor_critic(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Set device and random seed
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Initialize networks using config parameters
    actor = ActorNetwork(
        in_dim=config['input_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        hidden_layers=2,  # You might want to add this to your config
        layer_norm=True,
        activation=nn.ELU,
        epsilon=config['actor_epsilon']
    ).to(device)
    
    critic = CriticNetwork(
        obs_dim=config['input_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        num_buckets=config['num_buckets']
    ).to(device)
    
    # Setup optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config['actor_critic_lr'])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config['actor_critic_lr'])
    
    # Training loop
    num_episodes = config['episode_num']
    batch_size = config['batch_size']
    
    actor_losses = []
    critic_losses = []
    
    for episode in range(num_episodes):
        # Generate synthetic episode
        states, rewards, dones = generate_synthetic_episode(config, batch_size)
        states, rewards, dones = states.to(device), rewards.to(device), dones.to(device)
        
        # Get actions and values
        with torch.no_grad():
            action_logits = actor(states)
            actions, _ = actor.sample_action(action_logits)
            values, risks = critic(states, F.one_hot(actions, config['action_dim']).float())
        
        # Compute returns (simplified version without GAE)
        returns = torch.zeros_like(rewards)
        next_return = torch.zeros(batch_size, 1).to(device)
        
        for t in reversed(range(config['horizon'])):
            returns[t] = rewards[t] + config['gamma'] * next_return * (1 - dones[t])
            next_return = returns[t]
        
        # Update critic
        critic_optimizer.zero_grad()
        _, current_risks = critic(states, F.one_hot(actions, config['action_dim']).float())
        critic_loss = F.mse_loss(current_risks, returns)
        critic_loss.backward()
        critic_optimizer.step()
        
        # Update actor
        actor_optimizer.zero_grad()
        action_logits = actor(states)
        log_probs, entropy, _ = actor.evaluate_actions(action_logits, actions)
        
        advantage = (returns - values).detach()
        actor_loss = -(log_probs * advantage).mean() - config['entropy_scale'] * entropy.mean()
        actor_loss.backward()
        actor_optimizer.step()
        
        # Store losses
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        
        if (episode + 1) % 100 == 0:  # Changed to print every 100 episodes since you have more episodes
            print(f"Episode {episode+1}")
            print(f"Actor Loss: {actor_losses[-1]:.4f}")
            print(f"Critic Loss: {critic_losses[-1]:.4f}")
            print("-------------------")

if __name__ == "__main__":
    config_path = "config.yml"  # Make sure this points to your config.yml file
    test_actor_critic(config_path)