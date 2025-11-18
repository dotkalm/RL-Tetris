from pufferlib import pufferl
import os
import torch
import numpy as np


def test_environment_rewards():
    """Test if the Tetris environment gives rewards"""
    print("\n" + "="*60)
    print("TESTING TETRIS ENVIRONMENT REWARDS")
    print("="*60)
    
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    config["env"]["num_envs"] = 1
    
    # Load environment using pufferl like in training
    vecenv = pufferl.load_env(env_name, config)
    
    # Get the first environment from vecenv
    env = vecenv.envs[0] if hasattr(vecenv, 'envs') else vecenv
    
    obs = env.reset()
    
    # Handle if reset returns tuple (obs, info)
    if isinstance(obs, tuple):
        obs = obs[0]
    
    total_reward = 0
    for step in range(500):
        action = env.action_space.sample()  # Random action
        
        # Handle both vectorized and single env returns
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        
        # Handle array rewards from vectorized env
        if isinstance(reward, (list, np.ndarray)):
            reward = float(reward[0])
        else:
            reward = float(reward)
            
        # Handle array done flags
        if isinstance(terminated, (list, np.ndarray)):
            terminated = bool(terminated[0])
        if isinstance(truncated, (list, np.ndarray)):
            truncated = bool(truncated[0])
        
        if reward != 0:
            print(f"Step {step}: action={action}, reward={reward}")
            total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {step}, total reward: {total_reward}")
            break
    
    if total_reward == 0:
        print("WARNING: No rewards received in 500 random steps!")
        print("The Tetris environment might not be giving rewards for line clears")
    
    vecenv.close()
    print("="*60 + "\n")
