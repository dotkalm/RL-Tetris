from pufferlib import pufferl
import os
import torch
import numpy as np
from config import get_config


def main():
    # Load shared configuration
    env_name, config = get_config()
    
    # Removed verbose config printing for cleaner output

    vecenv = pufferl.load_env(env_name, config)
    policy = pufferl.load_policy(config, vecenv)
    
    train_config = dict(**config["train"], env=env_name)

    trainer = pufferl.PuffeRL(train_config, vecenv, policy)
    
    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {config['train']['total_timesteps']:,}")
    print(f"  Num environments: {vecenv.num_envs if hasattr(vecenv, 'num_envs') else 'unknown'}")
    print(f"  Total epochs: {trainer.total_epochs}")
    print(f"  Device: {config['train']['device']}\n") 

    if trainer.total_epochs == 0:
        print("ERROR: total_epochs is 0")
        return
    
    print("Training started...\n")
    try:
        while trainer.epoch < trainer.total_epochs:
            logs = trainer.train()

            if logs and trainer.epoch % 25 == 0:
                print(f"Epoch {trainer.epoch}/{trainer.total_epochs} - SPS: {logs.get('SPS', 0):.0f}")
            
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