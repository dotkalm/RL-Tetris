from pufferlib.ocean.tetris import tetris
from pufferlib import pufferl
import torch
import numpy as np

def evaluate_model(model_path, n_episodes=100, render=False):
    """Evaluate a trained model"""
    
    # Load config and create environment
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    config["train"]["device"] = "mps"
    config["train"]["use_rnn"] = False
    config["train"]["optimizer"] = "adam"
    
    # Create vectorized environment (same as training)
    config["env"]["num_envs"] = 1
    vecenv = pufferl.load_env(env_name, config)
    
    # Get the actual environment from vecenv (same action space as training)
    env = vecenv.envs[0] if hasattr(vecenv, 'envs') else vecenv
    
    # Create policy
    policy = pufferl.load_policy(config, vecenv)
    
    # Load trained weights
    policy.load_state_dict(torch.load(model_path, map_location='mps'))
    policy.eval()
    
    device = torch.device("mps")
    policy = policy.to(device)
    
    print(f"\nEvaluating model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print(f"Action space: {env.action_space}")
    print()
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Initialize LSTM state with size 512
        state = {
            'lstm_h': torch.zeros(1, 1, 128).to(device),
            'lstm_c': torch.zeros(1, 1, 128).to(device)
        }

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = policy(obs_tensor, state)
                
                # Debug output structure
                if episode == 0 and episode_length == 0:
                    print(f"Output type: {type(output)}")
                    if isinstance(output, tuple):
                        print(f"Output length: {len(output)}")
                        for i, item in enumerate(output):
                            if hasattr(item, 'shape'):
                                print(f"  Output[{i}] shape: {item.shape}")
                
                # Extract logits and new state
                if isinstance(output, tuple) and len(output) >= 1:
                    logits = output[0]
                    # Update state if returned
                    if len(output) >= 3:
                        state = output[2]
                else:
                    logits = output
                
                # Debug logits shape
                if episode == 0 and episode_length == 0:
                    print(f"Logits shape: {logits.shape}")
                    print(f"Action space: {env.action_space}")
                    print(f"Action space nvec: {env.action_space.nvec}")
                
                # Handle MultiDiscrete action space
                nvec = env.action_space.nvec
                num_discrete_actions = len(nvec)
                
                # Check if logits already have the right shape [num_actions, options_per_action]
                if logits.shape[0] == num_discrete_actions and logits.shape[1] == nvec[0]:
                    # Logits are already [num_actions, options_per_action]
                    action = torch.argmax(logits, dim=-1).cpu().numpy()  # [num_actions]
                elif len(logits.shape) == 3:  # [batch, num_actions, num_options]
                    action = torch.argmax(logits, dim=-1).cpu().numpy()
                    action = action[0]  # Remove batch dimension
                elif len(logits.shape) == 2:
                    # Could be [batch, total_logits] or [num_actions, options]
                    if logits.shape[1] == sum(nvec):  # [batch, total_logits]
                        batch_size = logits.shape[0]
                        # Reshape to [batch, num_actions, options]
                        logits_reshaped = logits.view(batch_size, num_discrete_actions, nvec[0])
                        action = torch.argmax(logits_reshaped, dim=-1).cpu().numpy()
                        action = action[0]
                    else:
                        # [batch, single_action_options] - single action
                        action = torch.argmax(logits, dim=-1).cpu().numpy()
                        action = action[0]
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Handle array rewards from vectorized env
            if isinstance(reward, (list, np.ndarray)):
                reward = float(reward[0])
            else:
                reward = float(reward)
            
            episode_reward += reward
            episode_length += 1
            
            # Handle array done flags
            if isinstance(terminated, (list, np.ndarray)):
                terminated = bool(terminated[0])
            if isinstance(truncated, (list, np.ndarray)):
                truncated = bool(truncated[0])
            
            # Debug first episode
            if episode == 0 and episode_length <= 10:
                print(f"Step {episode_length}: action={action}, reward={reward:.3f}")
            
            if render:
                import time
                env.render()
                time.sleep(0.02)
            
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Length: {episode_length}, Reward: {episode_reward:.2f}")
    
    env.close()
    vecenv.close()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Length:  {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Max Length:   {np.max(episode_lengths)}")
    print(f"Min Length:   {np.min(episode_lengths)}")
    print(f"Mean Reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Max Reward:   {np.max(episode_rewards):.2f}")
    print(f"Min Reward:   {np.min(episode_rewards):.2f}")
    print(f"Unique episode lengths: {len(set(episode_lengths))}/{n_episodes}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    model_path = "models/mlp/final_model.pt"
    n_episodes = 300
    render = False
    
    evaluate_model(model_path, n_episodes, render)