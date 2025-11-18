"""Simple test to verify Tetris environment works"""
from pufferlib import pufferl
from config import get_config

def test_single_episode():
    """Test if we can run a single episode"""
    env_name, config = get_config()
    
    print("Creating environment...")
    vecenv = pufferl.load_env(env_name, config)
    
    print(f"Num envs: {vecenv.num_envs if hasattr(vecenv, 'num_envs') else 'unknown'}")
    print(f"Action space: {vecenv.single_action_space}")
    print(f"Observation space: {vecenv.single_observation_space}")
    
    print("\nResetting environment...")
    obs = vecenv.reset()
    print(f"Obs shape: {obs.shape if hasattr(obs, 'shape') else 'not array'}")
    
    print("\nRunning 100 steps with random actions...")
    total_reward = 0
    done_count = 0
    
    for step in range(100):
        # Sample random actions for all environments
        actions = vecenv.action_space.sample()
        
        obs, rewards, terminated, truncated, infos = vecenv.step(actions)
        dones = terminated | truncated  # Combine terminated and truncated
        
        # Count rewards
        if hasattr(rewards, '__iter__'):
            total_reward += sum(rewards)
            done_count += sum(dones)
        else:
            total_reward += rewards
            done_count += int(dones)
        
        if step % 10 == 0:
            print(f"Step {step}: rewards sum={sum(rewards) if hasattr(rewards, '__iter__') else rewards}, dones={sum(dones) if hasattr(dones, '__iter__') else dones}")
    
    print(f"\nTotal reward over 100 steps: {total_reward}")
    print(f"Episodes completed: {done_count}")
    print(f"Average reward per step: {total_reward / 100:.3f}")
    
    vecenv.close()

if __name__ == "__main__":
    test_single_episode()
