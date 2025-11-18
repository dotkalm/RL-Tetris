"""
Evaluate trained CleanRL PPO agent on Tetris
"""
import torch
import numpy as np
import time
from train_ppo import PPOAgent
from pufferlib import pufferl
from pufferlib.ocean.tetris import tetris


def evaluate(model_path, n_episodes=10, render=False, delay=0.05):
    """Evaluate trained agent"""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create environment - use direct env when rendering for proper visualization
    if render:
        # Use single environment with render mode
        from gymnasium import spaces
        env = tetris.Tetris(render_mode='human')

        # Wrap in a simple object to match vecenv interface
        class SingleEnvWrapper:
            def __init__(self, env):
                self.env = env
                self.single_observation_space = env.observation_space
                # Convert MultiDiscrete([7]) to Discrete(7) to match trained agent
                if isinstance(env.action_space, spaces.MultiDiscrete):
                    self.single_action_space = spaces.Discrete(env.action_space.nvec[0])
                else:
                    self.single_action_space = env.action_space

            def reset(self):
                obs, info = self.env.reset()
                return np.array([obs]), [info]

            def step(self, actions):
                # Convert scalar action to array for MultiDiscrete space
                action = np.array([actions[0]]) if isinstance(self.env.action_space, spaces.MultiDiscrete) else actions[0]
                obs, reward, terminated, truncated, info = self.env.step(action)
                return np.array([obs]), np.array([reward]), np.array([terminated]), np.array([truncated]), [info]

            def render(self):
                return self.env.render()

            def close(self):
                self.env.close()

        vecenv = SingleEnvWrapper(env)
    else:
        # Use vectorized environment for faster evaluation
        env_name = "puffer_tetris"
        config = pufferl.load_config(env_name)
        config["vec"]["num_envs"] = 1
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

        # Render and add delay when rendering to make it visible
        if render:
            vecenv.render()
            time.sleep(delay)

        # Extract scalar values from arrays
        reward_scalar = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
        terminated_scalar = bool(terminated[0]) if isinstance(terminated, np.ndarray) else bool(terminated)
        truncated_scalar = bool(truncated[0]) if isinstance(truncated, np.ndarray) else bool(truncated)

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
    import argparse

    # Create parser that won't interfere with pufferlib's argument parsing
    parser = argparse.ArgumentParser(description='Evaluate Tetris agent', add_help=False)
    parser.add_argument('--render-mode', type=str, default=None, help='Render mode (human or None)')
    parser.add_argument('--episodes', type=int, default=40, help='Number of episodes')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay between steps when rendering')

    args, unknown = parser.parse_known_args()

    model_path = "../models/cleanrl/tetris_ppo.pt"
    render = args.render_mode == "human"
    n_episodes = args.episodes
    delay = args.delay

    evaluate(model_path, n_episodes=n_episodes, render=render, delay=delay)
