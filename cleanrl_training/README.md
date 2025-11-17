# CleanRL PPO Training for Tetris

This directory contains a CleanRL-style PPO implementation that actually works with the Tetris environment.

## Why This Works (vs PufferLib Trainer)

**PufferLib Trainer Issue:**
- `PuffeRL.train()` was collecting 0 steps from the environment
- Dashboard showed `Steps: 0`, `SPS: 0`, `Env: 0s 0%`
- Environment never actually ran despite being correctly configured

**CleanRL Solution:**
- Manual rollout collection with explicit `vecenv.step()` calls
- Transparent training loop you can debug and verify
- Proven PPO algorithm from CleanRL's battle-tested implementation

## Files

- `train_ppo.py` - Main training script with PPO algorithm
- `evaluate.py` - Evaluation script for trained models

## Usage

### Train

```bash
cd cleanrl_training
python train_ppo.py
```

Training will:
- Run for 1M timesteps
- Use 8 parallel environments
- Collect 128 steps per environment per rollout
- Save model to `../models/cleanrl/tetris_ppo.pt`
- Print progress every 10 updates

### Evaluate

```bash
python evaluate.py
```

To see the agent play with rendering:

```bash
python evaluate.py --render
```

## Hyperparameters

Current settings in `train_ppo.py`:

```python
total_timesteps = 1_000_000
num_envs = 8
num_steps = 128
learning_rate = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.2
ent_coef = 0.01
```

## Expected Training Time

With 8 parallel environments collecting 128 steps each:
- Batch size: 1024 steps
- ~977 updates for 1M timesteps
- Estimated time: 10-15 minutes on M2 Max

## Architecture

Simple MLP policy:
- Input: Flattened observation
- Hidden: 256 → ReLU → 256 → ReLU
- Actor head: Linear(256, 7) for actions
- Critic head: Linear(256, 1) for value

## Key Differences from PufferLib Attempt

1. **Explicit environment stepping**: We call `vecenv.step()` ourselves
2. **Manual rollout storage**: We store observations, actions, rewards, etc. in tensors
3. **Transparent GAE computation**: We calculate advantages explicitly
4. **Clear PPO updates**: We can see exactly what's being optimized

This gives us full control and visibility into the training process.
