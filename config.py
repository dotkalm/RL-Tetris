"""Shared configuration for training and evaluation"""
from pufferlib import pufferl


def get_config():
    """Get the standard configuration for training and evaluation"""
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    
    # Set number of parallel environments
    # vec.num_envs controls PufferLib's vectorization (number of parallel workers)
    # For M2 Max with 12 cores, use fewer workers to avoid oversubscription
    config["vec"]["num_envs"] = 8  # 8 parallel environments
    
    # Remove env.num_envs - Tetris doesn't use this parameter
    # Each environment runs a single Tetris game with Discrete(7) action space
    if "num_envs" in config["env"]:
        del config["env"]["num_envs"]
    
    # Core configuration
    config["train"]["device"] = "mps"
    config["train"]["total_timesteps"] = 10_000
    config["train"]["use_rnn"] = False
    config["train"]["learning_rate"] = 1e-4
    config["train"]["checkpoint_interval"] = 100

    config["train"]["optimizer"] = "adam"

    # Batch configuration
    # Reduced batch size for debugging - with 8 envs, this means 128 steps per env per batch
    config["train"]["batch_size"] = 1024
    config["train"]["minibatch_size"] = 256
    config["train"]["bptt_horizon"] = 16
    config["train"]["update_epochs"] = 4

    # PPO hyperparameters
    # Increased entropy coefficient from 0.01 to 0.05 to encourage exploration
    # This helps the agent discover rotation actions which don't give immediate rewards
    config["train"]["ent_coef"] = 0.05  # Higher entropy = more exploration of all actions

    # Entropy annealing: gradually decrease entropy over training for curriculum learning
    # Start high to explore rotation, then decrease to exploit learned strategy
    config["train"]["anneal_entropy"] = True
    config["train"]["ent_coef_final"] = 0.01  # End value

    config["train"]["vf_coef"] = 0.5
    config["train"]["max_grad_norm"] = 0.5
    config["train"]["clip_coef"] = 0.2
    config["train"]["gamma"] = 0.99
    config["train"]["gae_lambda"] = 0.95

    return env_name, config
