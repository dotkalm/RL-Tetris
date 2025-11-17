RL Tetris with pufferlib and stable_baselines3?
# Evaluate MLP (100 episodes, no rendering)
python evaluate.py --model models/mlp/final_model --episodes 100

# Watch MLP play (with rendering, slower)
python evaluate.py --model models/mlp/final_model --episodes 5 --render --sleep 0.05

# Evaluate CNN
python evaluate.py --model models/cnn/final_model --episodes 100
