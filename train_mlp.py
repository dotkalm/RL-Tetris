from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from pufferlib.ocean.tetris import tetris
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RL tetris')
    
    parser.add_argument(
        '--timesteps', 
        type=int, 
        default=500_000,
        help='timesteps | default: 500000'
    )

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("models/mlp", exist_ok=True)

    env = tetris.Tetris()
    eval_env = tetris.Tetris()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    print(f"number of timesteps: {args.timesteps}")

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/mlp/checkpoints/",
        name_prefix="tetris_mlp"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/mlp/",
        log_path="models/mlp/logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    model = PPO(
        'MlpPolicy',
        env,
        device="mps",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/mlp/"
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    model.save("models/mlp/final_model")
    print("Model saved to models/mlp/final_model")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()