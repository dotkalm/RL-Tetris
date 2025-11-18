"""
Watch the trained Tetris agent play with visual rendering
"""
import torch
import numpy as np
import time
from train_ppo import PPOAgent
from pufferlib import pufferl


def watch_agent(model_path, n_episodes=5, delay=0.1):
    """Watch trained agent play Tetris"""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create environment with rendering enabled
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    config["vec"]["num_envs"] = 1
    config["render_mode"] = "human"  # Enable rendering
    
    vecenv = pufferl.load_env(env_name, config)
    
    print(f"Note: Created {vecenv.num_envs} environments (PufferLib may create more than requested)")
    
    print(f"\n🎮 Watching Tetris Agent")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Delay per action: {delay}s")
    print(f"Note: Watching first environment only")
    print(f"{'='*60}\n")
    
    # Load agent
    agent = PPOAgent(vecenv.single_observation_space, vecenv.single_action_space).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    for episode in range(n_episodes):
        obs, _ = vecenv.reset()
        obs = torch.Tensor(obs).to(device)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n🎮 Episode {episode + 1}/{n_episodes}")
        
        while not done:
            # Get action from agent
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
                action = action.cpu().numpy()
            
            # Step environment (vecenv returns arrays)
            obs, reward, terminated, truncated, info = vecenv.step(action)
            obs = torch.Tensor(obs).to(device)
            
            # Watch only first environment
            done = terminated[0] or truncated[0]
            episode_reward += reward[0]
            episode_length += 1
            
            # Delay to make it watchable
            time.sleep(delay)
            
            # Print progress every 20 steps
            if episode_length % 20 == 0:
                print(f"  Step {episode_length}: Reward so far = {episode_reward:.2f}")
        
        print(f"  ✅ Episode complete!")
        print(f"  Final length: {episode_length}")
        print(f"  Final reward: {episode_reward:.2f}")
        print(f"  {'='*60}")
        
        # Pause between episodes
        if episode < n_episodes - 1:
            print("\n  Press Enter for next episode...")
            input()
    
    vecenv.close()
    print(f"\n✅ Finished watching {n_episodes} episodes!")


if __name__ == "__main__":
    import sys
    
    model_path = "../models/cleanrl/tetris_ppo.pt"
    
    # Parse optional arguments
    n_episodes = 5
    delay = 0.1
    
    for arg in sys.argv[1:]:
        if arg.startswith("--episodes="):
            n_episodes = int(arg.split("=")[1])
        elif arg.startswith("--delay="):
            delay = float(arg.split("=")[1])
    
    watch_agent(model_path, n_episodes=n_episodes, delay=delay)
