"""
Inspect the PufferLib Tetris environment to understand:
1. What the 234 observation dimensions represent
2. What the 7 actions do
3. The actual reward structure
"""
import pufferlib
from pufferlib import pufferl
import inspect
import numpy as np


def inspect_tetris_environment():
    """Inspect the Tetris environment structure"""
    
    print("="*60)
    print("INSPECTING PUFFERLIB TETRIS ENVIRONMENT")
    print("="*60)
    
    # Create environment - try to force 1 env
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    config["vec"]["num_envs"] = 1
    config["vec"]["num_workers"] = 1
    
    vecenv = pufferl.load_env(env_name, config)
    
    # Check how many we actually got
    actual_num_envs = vecenv.num_envs
    print(f"\nRequested 1 env, actually got: {actual_num_envs}")
    
    print(f"\n1. OBSERVATION SPACE")
    print(f"{'='*60}")
    print(f"Space: {vecenv.single_observation_space}")
    print(f"Shape: {vecenv.single_observation_space.shape}")
    print(f"Total dimensions: {np.prod(vecenv.single_observation_space.shape)}")
    
    # Get actual observation
    obs, _ = vecenv.reset()
    print(f"\nActual observation shape: {obs.shape}")
    print(f"Min value: {obs.min():.3f}, Max value: {obs.max():.3f}")
    print(f"\nFirst 30 values: {obs[0][:30]}")
    print(f"Last 30 values: {obs[0][-30:]}")
    
    # Try to infer structure
    print(f"\nInferred structure (guessing):")
    print(f"  - If 20x10 grid (200 values): cells 0-199")
    print(f"  - If 7-piece one-hot (7 values): cells 200-206")
    print(f"  - If next piece (7 values): cells 207-213")
    print(f"  - Remaining features: cells 214-233")
    
    print(f"\n2. ACTION SPACE")
    print(f"{'='*60}")
    print(f"Space: {vecenv.single_action_space}")
    print(f"Number of actions: {vecenv.single_action_space.n}")
    
    # Try taking different actions to see what happens
    print(f"\nTesting each action effect (for {actual_num_envs} envs):")
    for action_idx in range(vecenv.single_action_space.n):
        obs, _ = vecenv.reset()
        obs_before = obs[0].copy()
        
        # Create action array for all environments
        actions = np.array([action_idx] * actual_num_envs)
        obs_after, reward, terminated, truncated, info = vecenv.step(actions)
        
        print(f"  Action {action_idx}: reward={reward[0]:.4f}, terminated={terminated[0]}, truncated={truncated[0]}")
    
    print(f"\n3. REWARD STRUCTURE")
    print(f"{'='*60}")
    print("Testing different scenarios to understand rewards...")
    
    # Play a few random steps and track rewards
    obs, _ = vecenv.reset()
    total_reward = 0
    rewards_seen = []
    
    for step in range(100):
        # Sample action for all environments
        actions = np.array([vecenv.single_action_space.sample() for _ in range(actual_num_envs)])
        obs, reward, terminated, truncated, info = vecenv.step(actions)
        
        reward_val = reward[0] if isinstance(reward, np.ndarray) else reward
        if reward_val != 0:
            rewards_seen.append(reward_val)
        
        total_reward += reward_val
        
        if terminated[0] or truncated[0]:
            break
    
    print(f"Played {step} steps")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Unique non-zero rewards: {set(rewards_seen)}")
    if rewards_seen:
        print(f"Most common reward: {max(set(rewards_seen), key=rewards_seen.count)}")
    
    print(f"\n4. ENVIRONMENT SOURCE")
    print(f"{'='*60}")
    
    # Try to find the source code
    try:
        from pufferlib.ocean import tetris
        print(f"Found tetris module: {tetris.__file__}")
        
        # Check for Env class
        if hasattr(tetris, 'Env'):
            env_class = tetris.Env
            print(f"\nEnv class: {env_class}")
            
            # Try to get source
            try:
                source = inspect.getsource(env_class)
                print(f"\nSource code length: {len(source)} characters")
                print(f"\nFirst 500 characters of source:")
                print(source[:500])
                
                # Look for reward-related code
                if 'reward' in source.lower():
                    print("\n\nFound 'reward' in source. Searching...")
                    lines = source.split('\n')
                    for i, line in enumerate(lines):
                        if 'reward' in line.lower() and not line.strip().startswith('#'):
                            print(f"Line {i}: {line}")
            except Exception as e:
                print(f"Could not get source: {e}")
        
        # List all attributes
        print(f"\nTetris module attributes:")
        for attr in dir(tetris):
            if not attr.startswith('_'):
                print(f"  {attr}")
                
    except ImportError as e:
        print(f"Could not import tetris module: {e}")
    
    print(f"\n{'='*60}")
    print("INSPECTION COMPLETE")
    print(f"{'='*60}")
    
    vecenv.close()


if __name__ == "__main__":
    inspect_tetris_environment()
