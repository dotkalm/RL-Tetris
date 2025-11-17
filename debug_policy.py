"""Debug what policy pufferlib creates"""
from pufferlib import pufferl
import torch
from config import get_config

def debug_policy():
    # Load shared configuration
    env_name, config = get_config()

    print("Creating environment...")
    vecenv = pufferl.load_env(env_name, config)

    env = vecenv.envs[0] if hasattr(vecenv, 'envs') else vecenv
    print(f"Action space: {env.action_space}")
    print(f"Action space nvec: {env.action_space.nvec}")
    print(f"Total actions: {len(env.action_space.nvec)}")

    print("\nCreating policy...")
    policy = pufferl.load_policy(config, vecenv)

    print(f"Policy type: {type(policy)}")
    if hasattr(policy, 'output_size'):
        print(f"Policy output size: {policy.output_size}")

    # Test forward pass
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, list):
        obs = obs[0]

    device = torch.device("mps")
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    state = {
        'lstm_h': torch.zeros(1, 1, 128).to(device),
        'lstm_c': torch.zeros(1, 1, 128).to(device)
    }

    print("\nTesting forward pass...")
    with torch.no_grad():
        output = policy(obs_tensor, state)
        if isinstance(output, tuple):
            logits = output[0]
            print(f"Logits shape: {logits.shape}")
            print(f"Expected shape for {len(env.action_space.nvec)} actions: [{len(env.action_space.nvec)}, 7]")
            
            if logits.shape[0] != len(env.action_space.nvec):
                print(f"\n❌ MISMATCH!")
                print(f"Policy outputs {logits.shape[0]} action(s)")
                print(f"Environment expects {len(env.action_space.nvec)} actions")
            else:
                print(f"\n✅ Shapes match!")

    vecenv.close()

if __name__ == '__main__':
    debug_policy()
