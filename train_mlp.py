from pufferlib import pufferl
import os
import torch


def main():
    env_name = "puffer_tetris"
    config = pufferl.load_config(env_name)
    
    # Hardcoded configuration
    config["train"]["device"] = "mps"
    config["train"]["total_timesteps"] = 10000
    config["train"]["use_rnn"] = False
    config["train"]["learning_rate"] = 3e-4
    config["train"]["checkpoint_interval"] = 50
    config["env"]["num_envs"] = 8

    config["train"]["optimizer"] = "adam"

    config["train"]["minibatch_size"] = 512
    config["train"]["max_minibatch_size"] = 2048 

    vecenv = pufferl.load_env(env_name, config)
    policy = pufferl.load_policy(config, vecenv)
    
    train_config = dict(**config["train"], env=env_name)
    train_config["batch_size"] = 8192

    trainer = pufferl.PuffeRL(train_config, vecenv, policy)
    
    # Debug: check training will actually run
    print(f"Current epoch: {trainer.epoch}")
    print(f"Total epochs: {trainer.total_epochs}")
    print(f"Will train: {trainer.epoch < trainer.total_epochs}")
    print("\n====================\nbegin\n") 

    if trainer.total_epochs == 0:
        print("ERROR: total_epochs is 0")
        return
    
    print("training ...\n")
    epoch = 0
    try:
        while trainer.epoch < trainer.total_epochs:
            """
            SKIP EVALUATION/ conflict with use_rnn=False

            if epoch % 10 == 0:
                trainer.evaluate()
            """

            logs = trainer.train()

            if logs and epoch % 5 == 0:
                reward = logs.get('reward', logs.get('mean_reward', 0))
                print(f"Epoch {trainer.epoch}/{trainer.total_epochs} | Reward: {reward:.2f}")
            epoch += 1
            
    except KeyboardInterrupt:
        print("\n====================\ninterrupted")
    
    print(f"\nTraining completed. Final epoch: {trainer.epoch}")

    os.makedirs("models/mlp", exist_ok=True)
    save_path = "models/mlp/final_model"
    torch.save(policy.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    
    trainer.print_dashboard()
    trainer.close()
    
if __name__ == "__main__":
    main()