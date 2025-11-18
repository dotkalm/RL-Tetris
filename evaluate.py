"""
Evaluate trained CleanRL PPO agent on Tetris
"""
import torch
import numpy as np
from train_ppo import PPOAgent
from pufferlib import pufferl


def evaluate(model_path, n_episodes=10, render=False):
    """Evaluate trained agent"""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create environment
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    config["vec"]["num_envs"] = 1
    
    if render:
        config["render_mode"] = "human"
    
    vecenv = pufferl.load_env(env_name, config)
    
    # Load agent
    agent = PPOAgent(vecenv.single_observation_space, vecenv.single_action_space).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"\n🎮 Evaluating Tetris Agent")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print(f"{'='*60}\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = vecenv.reset()
        obs = torch.Tensor(obs).to(device)
        
        episode_reward = 0
        episode_length = 0
        done = False
        max_steps = 10000  # Prevent infinite episodes
        
        while not done and episode_length < max_steps:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
            
            obs, reward, terminated, truncated, info = vecenv.step(action.cpu().numpy())
            obs = torch.Tensor(obs).to(device)
            
            reward = reward[0] if isinstance(reward, np.ndarray) else reward
            done = (terminated[0] if isinstance(terminated, np.ndarray) else terminated) or \
                   (truncated[0] if isinstance(truncated, np.ndarray) else truncated)
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        timeout_msg = " (TIMEOUT)" if episode_length >= max_steps else ""
        print(f"Episode {episode + 1}/{n_episodes} - Length: {episode_length}, Reward: {episode_reward:.2f}{timeout_msg}")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"{'='*60}\n")
    
    vecenv.close()


if __name__ == "__main__":
    import sys
    
    model_path = "../models/cleanrl/tetris_ppo.pt"
    
    # Check for render flag and number of episodes
    render = "--render" in sys.argv
    n_episodes = 100  # Increased for better statistics
    
    # Allow custom episode count via command line
    for arg in sys.argv:
        if arg.startswith("--episodes="):
            n_episodes = int(arg.split("=")[1])
    
    evaluate(model_path, n_episodes=n_episodes, render=render)
