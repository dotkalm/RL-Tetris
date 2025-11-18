"""
CleanRL-style PPO implementation for Tetris
Based on CleanRL's PPO algorithm with vectorized environments
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from pufferlib import pufferl
import time
import os


class PPOAgent(nn.Module):
    """Simple MLP policy for Tetris"""
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        # Get dimensions
        if hasattr(obs_space, 'shape'):
            obs_dim = np.prod(obs_space.shape)
        else:
            obs_dim = obs_space.n
        
        action_dim = action_space.n
        
        # Shared network
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(256, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(256, 1)
    
    def get_value(self, x):
        """Get state value"""
        x = x.flatten(start_dim=1)
        hidden = self.network(x)
        return self.critic(hidden)
    
    def get_action_and_value(self, x, action=None):
        """Get action, log prob, entropy, and value"""
        x = x.flatten(start_dim=1)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def train_ppo():
    """Train Tetris agent with PPO"""
    
    # Hyperparameters
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    total_timesteps = 100_000
    
    # Start with 1 env for now due to PufferLib configuration issues
    # PufferLib keeps creating 4096 envs regardless of config
    num_envs = 1
    num_steps = 2048  # Larger rollouts to compensate for single env
    num_minibatches = 4
    update_epochs = 4
    
    learning_rate = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.02
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # Calculated values
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size
    
    # Create environment
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    
    # Force the number of environments - PufferLib seems to ignore this
    # Try both vec and env sections
    config["vec"]["num_envs"] = num_envs
    config["vec"]["num_workers"] = num_envs
    
    # Remove any auto settings
    if "batch_size" in config["vec"] and config["vec"]["batch_size"] == "auto":
        config["vec"]["batch_size"] = num_envs
    
    vecenv = pufferl.load_env(env_name, config)
    
    # Check if we got the right number of environments
    if vecenv.num_envs != num_envs:
        print(f"ERROR: PufferLib created {vecenv.num_envs} environments instead of {num_envs}")
        print("This is a known PufferLib configuration issue.")
        print("Adjusting training parameters to match...")
        num_envs = vecenv.num_envs
    
    print(f"Environment created:")
    print(f"  Action space: {vecenv.single_action_space}")
    print(f"  Observation space: {vecenv.single_observation_space}")
    print(f"  Num envs: {vecenv.num_envs}\n")
    
    global_step = 0
    start_time = time.time()
    next_obs, _ = vecenv.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    
    actual_num_envs = vecenv.num_envs
    
    if actual_num_envs != num_envs:
        num_envs = actual_num_envs
        batch_size = num_envs * num_steps
        minibatch_size = batch_size // num_minibatches
        num_updates = total_timesteps // batch_size
    
    next_done = torch.zeros(num_envs).to(device)
    
    actual_obs_shape = next_obs.shape[1:]
    
    agent = PPOAgent(vecenv.single_observation_space, vecenv.single_action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    obs_storage = torch.zeros((num_steps, num_envs) + actual_obs_shape).to(device)
    actions_storage = torch.zeros((num_steps, num_envs)).to(device)
    logprobs_storage = torch.zeros((num_steps, num_envs)).to(device)
    rewards_storage = torch.zeros((num_steps, num_envs)).to(device)
    dones_storage = torch.zeros((num_steps, num_envs)).to(device)
    values_storage = torch.zeros((num_steps, num_envs)).to(device)

    episode_rewards = []
    episode_lengths = []

    # Track action usage for rotation bonus
    action_counts = torch.zeros(7).to(device)  # 7 actions
    rotation_bonus_coef = 0.02  # Bonus coefficient for using rotation
    
    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            global_step += num_envs
            obs_storage[step] = next_obs
            dones_storage[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_storage[step] = value.flatten()
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            # Track action usage
            for a in action.cpu().numpy():
                action_counts[int(a)] += 1

            next_obs, reward, terminated, truncated, info = vecenv.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # Add rotation bonus: if action is ROTATE (3), give small bonus
            reward_tensor = torch.tensor(reward).to(device).view(-1)
            for i, a in enumerate(action):
                if a == 3:  # ACTION_ROTATE
                    reward_tensor[i] += rotation_bonus_coef

            rewards_storage[step] = reward_tensor
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)
            
            if "episode" in info:
                for item in info["episode"]:
                    if item is not None:
                        episode_rewards.append(item["r"])
                        episode_lengths.append(item["l"])
        
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_storage).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    nextvalues = values_storage[t + 1]
                delta = rewards_storage[t] + gamma * nextvalues * nextnonterminal - values_storage[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_storage
        
        # Flatten batch
        b_obs = obs_storage.reshape((-1,) + vecenv.single_observation_space.shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)
        
        # Optimize policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
        
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)

            # Calculate action distribution
            action_dist = action_counts / action_counts.sum()
            rotation_pct = action_dist[3].item() * 100  # ACTION_ROTATE = 3

            print(f"Update {update}/{num_updates}")
            print(f"  Global step: {global_step:,}")
            print(f"  SPS: {sps}")
            print(f"  Policy loss: {pg_loss.item():.4f}")
            print(f"  Value loss: {v_loss.item():.4f}")
            print(f"  Entropy: {entropy_loss.item():.4f}")
            print(f"  Rotation usage: {rotation_pct:.1f}%")

            if len(episode_rewards) > 0:
                print(f"  Episode reward (mean): {np.mean(episode_rewards[-100:]):.2f}")
                print(f"  Episode length (mean): {np.mean(episode_lengths[-100:]):.1f}")
            print()

            # Reset action counts for next period
            action_counts.zero_()
    
    # Save model
    os.makedirs("../models/cleanrl", exist_ok=True)
    model_path = "../models/cleanrl/tetris_ppo.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"\n✅ Training complete! Model saved to {model_path}")
    
    if len(episode_rewards) > 0:
        print(f"\n📊 Final Statistics:")
        print(f"  Total episodes: {len(episode_rewards)}")
        print(f"  Mean episode reward: {np.mean(episode_rewards):.2f}")
        print(f"  Mean episode length: {np.mean(episode_lengths):.1f}")
    
    vecenv.close()


if __name__ == "__main__":
    train_ppo()
