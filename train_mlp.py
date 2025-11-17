from pufferlib import pufferl
import os
import torch
import numpy as np
from test_environment_rewards import test_environment_rewards


def main():
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    
    print("Testing environment first...")
    test_environment_rewards()
    
    # Core configuration
    config["train"]["device"] = "mps"
    config["train"]["total_timesteps"] = 1_000_000  # More timesteps for learning
    config["train"]["use_rnn"] = False
    config["train"]["learning_rate"] = 1e-4  # Lower LR for stability
    config["train"]["checkpoint_interval"] = 100
    config["env"]["num_envs"] = 16  # Fewer envs for stability

    config["train"]["optimizer"] = "adam"

    # Batch configuration - batch_size must be > (num_envs * bptt_horizon)
    # With 16 envs: 16 * 32 = 512 total agents
    config["train"]["batch_size"] = 2048  # Must be > 512
    config["train"]["minibatch_size"] = 256
    config["train"]["bptt_horizon"] = 32
    config["train"]["update_epochs"] = 4

    # PPO hyperparameters - more conservative
    config["train"]["ent_coef"] = 0.01  # Standard entropy coefficient
    config["train"]["vf_coef"] = 0.5
    config["train"]["max_grad_norm"] = 0.5
    config["train"]["clip_coef"] = 0.2
    config["train"]["gamma"] = 0.99
    config["train"]["gae_lambda"] = 0.95


    vecenv = pufferl.load_env(env_name, config)
    policy = pufferl.load_policy(config, vecenv)
    
    train_config = dict(**config["train"], env=env_name)

    trainer = pufferl.PuffeRL(train_config, vecenv, policy)
    
    print(f"Total epochs: {trainer.total_epochs}")
    print(f"Device config: {config['train']['device']}")
    print(f"Policy device: {next(policy.parameters()).device}")
    print(f"Num envs: {config['env']['num_envs']}")
    print("\n====================\nbegin\n") 

    if trainer.total_epochs == 0:
        print("ERROR: total_epochs is 0")
        return
    
    print("training ...\n")
    try:
        while trainer.epoch < trainer.total_epochs:
            logs = trainer.train()

            if logs and trainer.epoch % 10 == 0:  # Use trainer.epoch
                print(f"\nEpoch {trainer.epoch}/{trainer.total_epochs}")
                print(f"SPS: {logs.get('SPS', 0):.0f}")
                print(f"Agent steps: {logs.get('agent_steps', 0)}")
                print(f"Policy loss: {logs.get('losses/policy_loss', 0):.4f}")
                print(f"Value loss: {logs.get('losses/value_loss', 0):.4f}")
                print(f"Entropy: {logs.get('losses/entropy', 0):.4f}")
            
    except KeyboardInterrupt:
        print("\n====================\ninterrupted")
    
    print(f"\nTraining completed. Final epoch: {trainer.epoch}")

    os.makedirs("models/mlp", exist_ok=True)
    save_path = "models/mlp/final_model.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    
    trainer.print_dashboard()
    trainer.close()
    
if __name__ == "__main__":
    main()