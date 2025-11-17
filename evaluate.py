from pufferlib.ocean.tetris import tetris
from pufferlib import pufferl
import torch
import numpy as np
from config import get_config

def evaluate_model(model_path, n_episodes=100, render=False):
    """Evaluate a trained model"""
    
    # Load shared configuration (same as training)
    env_name, config = get_config()
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
        try:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
            
            # Handle array obs from vecenv
            if isinstance(obs, (list, np.ndarray)) and len(np.array(obs).shape) > 1:
                obs = obs[0]
                
            done = False
            episode_reward = 0
            episode_length = 0
            
            # Initialize LSTM state
            state = {
                'lstm_h': torch.zeros(1, 1, 128).to(device),
                'lstm_c': torch.zeros(1, 1, 128).to(device)
            }

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = policy(obs_tensor, state)
                    
                    # Extract logits
                    if isinstance(output, tuple) and len(output) >= 1:
                        logits = output[0]
                    else:
                        logits = output
                    
                    # Handle MultiDiscrete action space
                    nvec = env.action_space.nvec
                    num_discrete_actions = len(nvec)
                    
                    # Check if logits already have the right shape
                    if logits.shape[0] == num_discrete_actions and logits.shape[1] == nvec[0]:
                        action = torch.argmax(logits, dim=-1).cpu().numpy()
                    elif len(logits.shape) == 2 and logits.shape[1] == sum(nvec):
                        batch_size = logits.shape[0]
                        logits_reshaped = logits.view(batch_size, num_discrete_actions, nvec[0])
                        action = torch.argmax(logits_reshaped, dim=-1).cpu().numpy()
                        action = action[0]
                    else:
                        action = torch.argmax(logits, dim=-1).cpu().numpy()
                        if isinstance(action, np.ndarray) and action.size == 1:
                            action = action[0]
                    
                    # Debug action before reshape
                    if episode == 0 and episode_length == 0:
                        print(f"Logits shape: {logits.shape}")
                        print(f"Num discrete actions: {num_discrete_actions}")
                        print(f"Action before reshape: {action}")
                        print(f"Action type: {type(action)}")
                        print(f"Action shape: {np.array(action).shape}")
                    
                    # Vectorized env expects action shape (num_discrete_actions, 1)
                    # Ensure action is always array of shape (num_discrete_actions,)
                    if not isinstance(action, np.ndarray):
                        action = np.array([action] * num_discrete_actions)
                    elif len(action.shape) == 0:  # scalar
                        action = np.array([int(action)] * num_discrete_actions)
                    elif action.shape[0] != num_discrete_actions:
                        # Action has wrong number of dimensions
                        print(f"WARNING: Action has {action.shape[0]} dims, expected {num_discrete_actions}")
                        action = np.array([int(action[0])] * num_discrete_actions)
                    
                    if episode == 0 and episode_length == 0:
                        print(f"Action after processing: {action}")
                        print(f"Action shape after processing: {action.shape}")
                    
                    # Now reshape to (num_discrete_actions, 1)
                    action = action.reshape(-1, 1)
                    
                    if episode == 0 and episode_length == 0:
                        print(f"Action final shape: {action.shape}")
                        print(f"Expected shape: ({num_discrete_actions}, 1)")
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Handle array rewards
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
                if episode == 0 and episode_length <= 5:
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
                      
        except Exception as e:
            print(f"Error during episode {episode}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next episode instead of breaking
            continue
    
    vecenv.close()
    
    # Print results
    if len(episode_rewards) > 0:
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Episodes completed: {len(episode_rewards)}/{n_episodes}")
        print(f"Mean Length:  {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Max Length:   {np.max(episode_lengths)}")
        print(f"Min Length:   {np.min(episode_lengths)}")
        print(f"Mean Reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Max Reward:   {np.max(episode_rewards):.2f}")
        print(f"Min Reward:   {np.min(episode_rewards):.2f}")
        print(f"Unique episode lengths: {len(set(episode_lengths))}/{len(episode_lengths)}")
        print(f"{'='*60}\n")
    else:
        print("No episodes completed successfully!")

if __name__ == "__main__":
    model_path = "models/mlp/final_model.pt"
    n_episodes = 10  # Start with just 10 to test
    render = False
    
    evaluate_model(model_path, n_episodes, render)
