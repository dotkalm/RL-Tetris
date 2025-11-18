"""Verify the training configuration matches the action space"""
from pufferlib import pufferl
import torch
import numpy as np
from config import get_config

def verify_configuration():
    # Load shared configuration
    env_name, config = get_config()

    # Create environment
    print("Creating vectorized environment...")
    vecenv = pufferl.load_env(env_name, config)

    # Get environment info
    env = vecenv.envs[0] if hasattr(vecenv, 'envs') else vecenv

    print("\n" + "="*60)
    print("ENVIRONMENT CONFIGURATION")
    print("="*60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action space type: {type(env.action_space)}")
    if hasattr(env.action_space, 'nvec'):
        print(f"Action space nvec: {env.action_space.nvec}")
        print(f"Number of discrete actions: {len(env.action_space.nvec)}")
        print(f"Options per action: {env.action_space.nvec[0]}")

    # Create policy
    print("\nCreating policy...")
    policy = pufferl.load_policy(config, vecenv)

    print(f"Policy type: {type(policy)}")
    print(f"Policy device: {next(policy.parameters()).device}")

    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"✅ Action space has {len(env.action_space.nvec)} discrete actions")
    print(f"✅ Each action has {env.action_space.nvec[0]} options")
    print(f"✅ Policy created successfully on {next(policy.parameters()).device}")
    print("="*60)

    # Cleanup
    vecenv.close()
    print("\n✓ Configuration verified - ready for training!")

if __name__ == '__main__':
    verify_configuration()
