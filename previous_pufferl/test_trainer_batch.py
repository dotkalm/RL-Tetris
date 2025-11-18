"""Test if PufferLib trainer can collect a single batch"""
from pufferlib import pufferl
from config import get_config
import torch

def test_trainer_batch():
    """Test collecting a single training batch"""
    env_name, config = get_config()
    
    print("Creating environment and policy...")
    vecenv = pufferl.load_env(env_name, config)
    policy = pufferl.load_policy(config, vecenv)
    
    print(f"Environment: {vecenv.num_envs} parallel envs")
    print(f"Batch size: {config['train']['batch_size']}")
    print(f"Steps per env per batch: {config['train']['batch_size'] / vecenv.num_envs}")
    
    train_config = dict(**config["train"], env=env_name)
    trainer = pufferl.PuffeRL(train_config, vecenv, policy)
    
    # Check if trainer's vecenv is accessible
    print(f"\nTrainer vecenv: {trainer.vecenv if hasattr(trainer, 'vecenv') else 'not accessible'}")
    print(f"Trainer has these attributes: {[a for a in dir(trainer) if not a.startswith('_')][:10]}")
    
    print(f"\nCollecting first batch...")
    try:
        logs = trainer.train()
        print(f"\nFirst batch collected!")
        print(f"  Agent steps: {logs.get('agent_steps', 0)}")
        print(f"  SPS: {logs.get('SPS', 0):.0f}")
        print(f"  Policy loss: {logs.get('losses/policy_loss', 0):.4f}")
        print(f"  Available keys: {list(logs.keys())}")
        
        # Check if there are any user stats
        user_stats = {k: v for k, v in logs.items() if not k.startswith('losses/') and not k.startswith('performance/')}
        print(f"\nUser stats: {user_stats}")
        
    except Exception as e:
        print(f"ERROR collecting batch: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.close()

if __name__ == "__main__":
    test_trainer_batch()
