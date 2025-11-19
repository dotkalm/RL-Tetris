# PPO Training for Tetris: Technical Walkthrough

## Overview
This codebase implements **Proximal Policy Optimization (PPO)** to train a reinforcement learning agent to play Tetris. The implementation follows CleanRL's explicit, transparent approach rather than using a black-box trainer.

---

## Architecture: The Neural Network (`PPOAgent`)

### Actor-Critic Architecture
The agent uses a **multi-layer perceptron (MLP)** with an **actor-critic** design:

```
Input (234 dimensions) → Shared Network → Two Heads:
                                          ├─ Actor (Policy)
                                          └─ Critic (Value Function)
```

### Components:

1. **Shared Network** (Feature Extractor)
   - Input: 234-dimensional observation (flattened Tetris game state)
   - Layer 1: Linear(234 → 256) + ReLU activation
   - Layer 2: Linear(256 → 256) + ReLU activation
   - Output: 256-dimensional hidden representation

2. **Actor Head** (Policy Network)
   - Linear(256 → 7) - outputs logits for 7 discrete actions
   - Produces a **stochastic policy**: probability distribution over actions
   - Actions: {0: move left, 1: move right, 2: rotate CW, 3: rotate CCW, 4: soft drop, 5: hard drop, 6: no-op}

3. **Critic Head** (Value Function)
   - Linear(256 → 1) - outputs a single scalar value
   - Estimates **state value V(s)**: expected cumulative reward from this state

### Why Actor-Critic?
- **Actor**: Learns what actions to take (policy optimization)
- **Critic**: Learns how good states are (reduces variance in policy gradients)
- **Shared layers**: Features useful for both policy and value estimation

---

## The PPO Algorithm

### Key Concepts

#### 1. **Rollout Collection Phase**
```python
for step in range(num_steps):  # 2048 steps
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    next_obs, reward, terminated, truncated, info = vecenv.step(action)
```

- **Rollout**: Collect `num_steps` (2048) timesteps of experience
- **Vectorized environments**: Run multiple Tetris games in parallel (PufferLib creates 1024 envs)
- **Stores**: observations, actions, log probabilities, rewards, values, done flags

**What happens in one step:**
1. Neural network observes current game state (234 dims)
2. Actor head outputs action probabilities, samples an action
3. Critic head estimates state value
4. Environment executes action → new state, reward
5. Store all data in rollout buffers

#### 2. **Advantage Estimation (GAE)**
```python
delta = reward + gamma * next_value * (1 - done) - current_value
advantage = delta + gamma * gae_lambda * (1 - done) * last_advantage
```

**Generalized Advantage Estimation (GAE)** answers: "How much better was this action than expected?"

- **Temporal Difference (TD) Error (δ)**: `reward + γ*V(next_state) - V(current_state)`
- **Advantage A(s,a)**: Exponentially-weighted sum of TD errors
- **Lambda (λ = 0.95)**: Controls bias-variance tradeoff
  - λ=0: Only immediate TD error (low variance, high bias)
  - λ=1: Full Monte Carlo returns (high variance, low bias)
- **Returns**: `R = A + V` (what we actually got vs. what critic predicted)

#### 3. **Policy Update Phase**
For `update_epochs` (4) epochs, iterate over `minibatches`:

**a) Policy Loss (Clipped Surrogate Objective)**
```python
ratio = exp(new_logprob - old_logprob)  # π_new / π_old
clipped_ratio = clip(ratio, 1-ε, 1+ε)   # ε = 0.2
pg_loss = -min(ratio * advantage, clipped_ratio * advantage)
```

- **Ratio**: How much did policy change?
- **Clipping**: Prevents policy from changing too drastically
- **Objective**: Maximize expected advantage, but not too aggressively

**b) Value Loss (MSE)**
```python
v_loss = 0.5 * (predicted_value - actual_return)²
```
- Trains critic to better predict returns
- Factor 0.5 (`vf_coef`) balances value loss vs. policy loss

**c) Entropy Bonus**
```python
entropy = -Σ p(a) * log p(a)
```
- **Encourages exploration**: Penalizes deterministic policies
- High entropy → agent explores diverse actions
- Low entropy → agent exploits known good actions
- Coefficient: `ent_coef = 0.03` (controls exploration strength)

**d) Total Loss**
```python
loss = pg_loss - ent_coef * entropy + vf_coef * v_loss
```
- Backpropagate through neural network
- Clip gradients to `max_grad_norm = 0.5` (stability)
- Update weights with Adam optimizer (`lr = 2.5e-4`)

---

## Training Loop Structure

### Hierarchical Organization

```
Training Run (500,000 timesteps)
│
├─ Update 1 (2048 steps × 1024 envs = 2,097,152 transitions)
│   ├─ Rollout Phase: Collect 2048 steps from 1024 parallel envs
│   ├─ Compute advantages with GAE
│   └─ Optimization Phase:
│       └─ Epoch 1, 2, 3, 4
│           └─ Minibatch 1, 2, 3, 4 (each ~524k transitions)
│               └─ Forward pass → Compute loss → Backward pass → Update weights
│
├─ Update 2
│   └─ (repeat above)
│
└─ Update ~244 (until 500k timesteps)
```

### Terminology Mapping

| Term | Meaning in This Code | Example Value |
|------|----------------------|---------------|
| **Timestep** | Single environment step (obs → action → reward) | 1 step in 1 env |
| **Step** | One rollout collection iteration across all envs | 2048 steps |
| **Global Step** | Total timesteps across all envs | `global_step += num_envs` per step |
| **Rollout** | Full sequence of steps before policy update | 2048 steps |
| **Minibatch** | Subset of rollout data for one gradient update | ~524k transitions |
| **Epoch** | One pass through entire rollout data | 4 epochs per update |
| **Update** | Complete optimization cycle (rollout → epochs → minibatches) | 244 updates total |
| **Episode** | One complete Tetris game (start → game over) | ~50-150 steps (varies) |

### Key Differences: Steps vs. Episodes

- **Training is organized by STEPS**, not episodes
- **Episodes end asynchronously** across parallel environments
- One update contains ~15-40 episodes (across 1024 parallel envs × 2048 steps)
- Episode statistics tracked via `info["episode"]` when episodes complete

---

## Hyperparameters Explained

### Learning Parameters
- **`learning_rate = 2.5e-4`**: Step size for gradient descent (standard PPO value)
- **`gamma = 0.99`**: Discount factor - values rewards 1 step ahead at 99% of current value
- **`gae_lambda = 0.95`**: GAE smoothing - balances bias vs. variance in advantage estimates

### PPO-Specific
- **`clip_coef = 0.2`**: Policy can change by max ±20% per update (prevents destructive updates)
- **`ent_coef = 0.03`**: Entropy bonus strength (increased from typical 0.01 for more exploration)
- **`vf_coef = 0.5`**: Value function loss weight (balance critic vs. actor training)
- **`max_grad_norm = 0.5`**: Gradient clipping threshold (numerical stability)

### Batch Configuration
- **`num_steps = 2048`**: Rollout length before updating policy
- **`num_minibatches = 4`**: Split batch into 4 chunks for memory efficiency
- **`update_epochs = 4`**: Reuse each batch 4 times (sample efficiency)
- **`minibatch_size`**: `batch_size / num_minibatches` = ~524k transitions

### Environment
- **`num_envs = 1`** (requested) but PufferLib creates **1024 envs**
- **`total_timesteps = 500,000`**: Training budget (you trained 1B in practice)

---

## Why CleanRL Style? (vs. PufferLib Trainer)

### The Problem with PufferLib
```python
# PufferLib's approach (BROKEN):
trainer = PuffeRL(env, agent, ...)
trainer.train()  # Black box - collected 0 steps!
```
- `PuffeRL.train()` never called `vecenv.step()`
- No visibility into what's happening
- Impossible to debug

### CleanRL's Solution (What You Use)
```python
# Explicit control:
for update in range(num_updates):
    # Rollout phase - YOU control the loop
    for step in range(num_steps):
        action = agent.get_action(obs)
        obs, reward = vecenv.step(action)  # EXPLICIT step call
        # Store data
    
    # Optimization phase - YOU implement PPO
    for epoch in range(update_epochs):
        for minibatch in minibatches:
            loss = compute_ppo_loss(minibatch)
            loss.backward()
            optimizer.step()
```

**Benefits:**
- ✅ Full transparency - see every step
- ✅ Easy to debug - add prints anywhere
- ✅ Actually works - you control `vecenv.step()`
- ✅ Flexible - modify algorithm easily

---

## Config.py vs. train_ppo.py

### `config.py` (LEGACY - Not Currently Used)
- Created for PufferLib's trainer interface
- Contains hyperparameters formatted for `PuffeRL.train()`
- **Status**: Superseded by inline hyperparameters in `train_ppo.py`

### `train_ppo.py` (ACTIVE)
- Hyperparameters defined directly in code
- Complete control over training loop
- Used for actual training

**Why the duplication?**
- Started with PufferLib approach (`config.py`)
- Switched to CleanRL when trainer broke
- `config.py` kept for reference but not imported

---

## Training Workflow Summary

1. **Initialize**: Create 1024 parallel Tetris environments + neural network
2. **Loop for 244 updates**:
   - **Collect**: Run 2048 steps in each environment → 2M transitions
   - **Process**: Compute advantages using GAE
   - **Optimize**: 
     - 4 epochs through data
     - 4 minibatches per epoch
     - 16 gradient updates per rollout
3. **Save**: Store trained neural network weights to `models/cleanrl/tetris_ppo.pt`

**Total**: ~500 million transitions, ~244 policy updates, ~3,904 gradient steps

---

## Key Machine Learning Concepts Used

| Concept | Implementation | Purpose |
|---------|---------------|---------|
| **On-Policy RL** | PPO collects data with current policy, updates, discards data | Sample efficiency |
| **Actor-Critic** | Separate policy (actor) and value (critic) heads | Reduce variance |
| **Trust Region** | Clipped ratio prevents large policy changes | Training stability |
| **GAE** | Weighted TD errors for advantage estimation | Bias-variance tradeoff |
| **Entropy Regularization** | Bonus for stochastic policies | Encourage exploration |
| **Minibatch SGD** | Split data into chunks for gradient updates | Memory efficiency |
| **Gradient Clipping** | Limit gradient magnitude | Numerical stability |
| **Vectorization** | Parallel environments for data collection | Computational efficiency |

---

## Results

After training:
- **Random agent**: ~50 steps/episode, ~2 reward
- **Trained agent**: ~122 steps/episode, ~6.6 reward
- **Improvement**: 2.4× longer survival, learned to use soft drop for survival bonus
- **Strategy**: Conservative play, prioritizes survival over aggressive line clearing

**Learned behaviors:**
- Uses action 4 (soft drop) frequently → +0.01 reward per step
- Occasionally uses action 5 (hard drop) for line clears → +0.3-0.4 reward
- Avoids risky moves that might end game quickly

---

## File Summary

- **`train_ppo.py`**: Complete PPO implementation (neural network + algorithm)
- **`evaluate.py`**: Load trained model, test on new episodes, report statistics
- **`config.py`**: Legacy hyperparameters (not actively used)
- **`models/cleanrl/tetris_ppo.pt`**: Saved neural network weights (657KB)
- **`basic.py`**: Simple Tetris environment test (renders game with random actions)
