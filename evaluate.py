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

    # Reset once at the start - vectorized env will auto-reset after each episode
    print("Resetting environment...")
    obs, _ = vecenv.reset()
    obs = torch.Tensor(obs).to(device)
    print(f"Environment ready. Observation shape: {obs.shape}")

    episodes_completed = 0
    max_steps_total = n_episodes * 10000  # Safety limit to prevent infinite loop
    steps = 0

    # Track current episode stats
    current_episode_reward = 0
    current_episode_length = 0

    print("Starting evaluation loop...\n")

    while episodes_completed < n_episodes and steps < max_steps_total:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)

        obs, reward, terminated, truncated, info = vecenv.step(action.cpu().numpy())
        obs = torch.Tensor(obs).to(device)

        # Extract scalar values from arrays
        reward_scalar = reward[0] if isinstance(reward, np.ndarray) else reward
        terminated_scalar = terminated[0] if isinstance(terminated, np.ndarray) else terminated
        truncated_scalar = truncated[0] if isinstance(truncated, np.ndarray) else truncated

        current_episode_reward += reward_scalar
        current_episode_length += 1
        steps += 1

        # Check if episode ended
        if terminated_scalar or truncated_scalar:
            episodes_completed += 1
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)

            timeout_msg = " (TIMEOUT)" if current_episode_length >= 10000 else ""
            print(f"Episode {episodes_completed}/{n_episodes} - Length: {current_episode_length}, Reward: {current_episode_reward:.2f}{timeout_msg}")

            # Reset tracking for next episode (env auto-resets)
            current_episode_reward = 0
            current_episode_length = 0

            if episodes_completed >= n_episodes:
                break
    
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
    n_episodes = 40  # Increased for better statistics
    
    # Allow custom episode count via command line
    for arg in sys.argv:
        if arg.startswith("--episodes="):
            n_episodes = int(arg.split("=")[1])
    
    evaluate(model_path, n_episodes=n_episodes, render=render)
