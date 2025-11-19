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

Deep Dive Explanations
GAE (Generalized Advantage Estimation)
Problem it solves: How do we know if an action was actually good or just got lucky?

The Challenge:

If you win Tetris after 1000 moves, which of those 1000 actions were actually good decisions?
Should we credit all 1000 equally? Only the last few? Something in between?
How GAE works:
Imagine you're at step 100 in a Tetris game. GAE asks: "Was taking that action better than what I normally expect from this position?"

It calculates this by looking forward in time multiple steps:

1-step ahead: Did I get immediate reward better than expected?
2-steps ahead: What about the reward after that?
3-steps ahead: And the one after that?
... up to end of episode
Then it exponentially weights these: nearby steps matter more (λ=0.95 means each step ahead is worth 95% of the previous).

Why it matters: Without GAE, the agent gets confused about which actions led to success. With GAE, it learns faster because credit is assigned more accurately.

MSE (Mean Squared Error)
What it is: A way to measure "how wrong" your predictions are.

The Math: (prediction - actual)²

Why square it?

Makes all errors positive (otherwise -5 and +5 would cancel out)
Punishes big errors MORE than small errors (error of 10 is 100x worse than error of 1)
Makes the math work nicely with calculus (smooth derivatives)
In your code: The critic network predicts "I think this Tetris position is worth 5.2 reward". After playing, you actually got 6.6 reward. MSE = (5.2 - 6.6)² = 1.96. The network learns to predict better next time.

Why "Mean"?: You average the squared errors across your entire minibatch (~524k predictions), so one bad prediction doesn't dominate.

Backpropagation
The Core Idea: Teaching the neural network by working backwards through its layers.

The Metaphor: Imagine a factory assembly line that produces wrong products. Backpropagation is like investigating which station screwed up by going backwards from the final product.

How it works:

Forward Pass: Input (234 dims) → Hidden (256) → Output (7 action probs + 1 value)
Calculate Loss: "The output was wrong by THIS much"
Backward Pass: Starting from the output, calculate:
How much did the last layer contribute to the error?
How much did the second-to-last layer contribute?
Keep going backward to the first layer
Update Weights: Adjust each layer's weights proportional to how much they contributed to the error
The Math: Uses calculus (chain rule) to compute gradients - the "slope" showing which direction to adjust each of the 60,000+ neural network parameters.

Why it's powerful: You don't have to manually figure out what each neuron should do. The algorithm automatically discovers which patterns in the Tetris grid predict good moves.

Adam Optimizer
What it does: Decides how much to adjust neural network weights during training.

The Problem with Simple Gradient Descent:

Learning rate too high: Network jumps around wildly, never converges
Learning rate too low: Training takes forever
Same learning rate for all parameters: Some need big updates, some need tiny tweaks
Adam's Solution (Adaptive Moment Estimation):

Momentum: Remembers which direction updates have been going recently. Like a ball rolling downhill - builds up speed in consistent directions.

Adaptive Learning Rates: Each of the 60,000+ parameters gets its own learning rate that adjusts automatically:

Parameters that change a lot → smaller learning rate (careful!)
Parameters that barely change → larger learning rate (move faster!)
Bias Correction: Accounts for the fact that at the start of training, the momentum estimates are unreliable.

Why it's popular: "Set it and forget it" - works well on most problems without hand-tuning. Your learning_rate=2.5e-4 is just a starting point; Adam adjusts from there.

Hyperparameters:

lr = 2.5e-4: Base learning rate (0.00025 - small steps)
beta1 = 0.9: How much to trust momentum (default)
beta2 = 0.999: How much to trust variance estimates (default)
eps = 1e-5: Numerical stability (prevents division by zero)
Buffer Filling & Update Cycles
"One update contains ~15-40 episodes" - What does this mean?
The Buffer:

You have 1024 parallel Tetris games running simultaneously
Each collects 2048 steps of experience
Total: 1024 × 2048 = 2,097,152 transitions stored in memory buffers
This is NOT about episodes:

Episodes are independent of the buffer size
An episode is one complete Tetris game (start → game over)
Average episode: ~50-150 steps
Buffer: 2,097,152 steps worth of data from all games combined
Why ~15-40 episodes per update?

1024 environments × average ~20-40 episode completions during the 2048 steps
Some envs finish 0 episodes (game still ongoing)
Some envs finish 3-4 episodes (died quickly, restarted multiple times)
The buffer fills based on STEPS, not episodes
Buffer lifecycle:

Empty buffers at start of update
Fill for 2048 steps (regardless of how many episodes finish)
Process all data through PPO optimization
Discard everything (on-policy learning - old data is stale)
Repeat with fresh buffer
loss.backward()
What it does: Triggers backpropagation through the entire neural network.

The Sequence:

What's happening inside .backward():

Computation Graph: PyTorch remembered every operation that led to the loss

"Loss came from policy_loss + value_loss"
"Policy_loss came from action probabilities"
"Action probabilities came from actor layer"
"Actor layer came from hidden layer"
"Hidden layer came from input"
Chain Rule: Working backwards, compute:

∂loss/∂(actor_weights) = how much does changing actor weights change loss?
∂loss/∂(hidden_weights) = how much does changing hidden weights change loss?
Do this for ALL 60,000+ parameters
Gradient Storage: Each parameter's .grad attribute now contains its gradient

Why separate .backward() from .step()?

.backward(): Calculate what to change (gradients)
.step(): Actually change it (update weights)
Separation allows gradient clipping, gradient accumulation, etc.
Hyperparameters Explained
What they are: Knobs you tune BEFORE training that control how learning happens. Unlike weights (learned during training), hyperparameters are set by you.

Categories:
1. Learning Rate Parameters

learning_rate = 2.5e-4: How big of steps to take when updating weights
Too high: Network becomes unstable, forgets what it learned
Too low: Training takes forever, might get stuck
2.5e-4 is standard for PPO
2. Discount Factor (γ = 0.99)

How much to value future rewards vs. immediate rewards
0.99 = "Reward 100 steps from now is worth 0.99^100 ≈ 37% of immediate reward"
Tetris needs high gamma because game is long-term strategic
3. GAE Lambda (λ = 0.95)

Bias-variance tradeoff in advantage estimation
High (0.95-0.99): Trust long-term outcomes (higher variance but less bias)
Low (0.8-0.9): Trust short-term signals (lower variance but more bias)
4. PPO Clip Coefficient (ε = 0.2)

Maximum allowed policy change per update
Ratio can only be between 0.8 and 1.2 (±20%)
Prevents catastrophic forgetting - agent can't suddenly become terrible
5. Entropy Coefficient (0.03)

How much to reward exploration
High (0.1): "Try random stuff, even if it seems bad"
Low (0.001): "Only do what you know works"
0.03 is medium - balance exploration vs. exploitation
6. Value Function Coefficient (0.5)

How much to weight critic loss vs. policy loss
1.0 = equal importance
0.5 = policy matters 2x more than value prediction
7. Batch Architecture

num_steps = 2048: Rollout length (how much data before updating)
num_minibatches = 4: Split data into chunks (memory efficiency)
update_epochs = 4: Reuse each batch 4 times (sample efficiency)
Why so many? Each controls a different aspect:

Network architecture (hidden size)
Optimization (learning rate, Adam betas)
RL algorithm (gamma, lambda, clip)
Training stability (gradient clipping, entropy)
Computational efficiency (batch sizes, epochs)
Tuning them: Art + science. Start with proven defaults (like CleanRL's), then adjust based on your specific problem. Your entropy is higher (0.03 vs typical 0.01) because Tetris needs more exploration to discover rotation strategies.