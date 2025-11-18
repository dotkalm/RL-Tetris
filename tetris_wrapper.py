"""
Tetris environment wrapper with reward shaping to encourage rotation
"""
import numpy as np
import gymnasium as gym


class RotationRewardWrapper(gym.Wrapper):
    """
    Wrapper that adds auxiliary rewards for rotation to encourage the agent to explore this action.

    Strategy:
    1. Small reward for any valid rotation (to encourage exploration)
    2. Bonus reward when rotation leads to line clears
    3. Track rotation usage to ensure the agent is learning to use it
    """

    def __init__(self, env, rotation_reward=0.01, rotation_usage_target=0.1):
        super().__init__(env)
        self.rotation_reward = rotation_reward
        self.rotation_usage_target = rotation_usage_target
        self.last_action = None
        self.episode_rotations = 0
        self.episode_steps = 0
        self.last_lines_cleared = 0

    def reset(self, **kwargs):
        self.episode_rotations = 0
        self.episode_steps = 0
        self.last_lines_cleared = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_steps += 1

        # ACTION_ROTATE = 3 (from tetris.h)
        if action == 3:
            self.episode_rotations += 1
            # Add small reward for rotation to encourage exploration
            reward += self.rotation_reward

            # Additional bonus if rotation led to line clears
            # (we detect this by seeing if reward is > 0.1, indicating a line clear)
            if reward > 0.1:  # Line clear happened
                reward += 0.02  # Bonus for rotating before clearing lines

        # Store for next step
        self.last_action = action

        # Add rotation statistics to info
        if terminated or truncated:
            if 'episode' not in info:
                info['episode'] = {}
            rotation_rate = self.episode_rotations / max(self.episode_steps, 1)
            info['episode']['rotation_rate'] = rotation_rate
            info['episode']['total_rotations'] = self.episode_rotations

        return obs, reward, terminated, truncated, info


class HeightPenaltyWrapper(gym.Wrapper):
    """
    Wrapper that adds reward shaping based on board height to encourage better piece placement.
    Lower height = better, which incentivizes clearing lines and using rotation effectively.
    """

    def __init__(self, env, height_penalty_scale=0.001):
        super().__init__(env)
        self.height_penalty_scale = height_penalty_scale
        self.last_height = 0

    def reset(self, **kwargs):
        self.last_height = 0
        return self.env.reset(**kwargs)

    def _get_board_height(self, obs):
        """
        Extract board height from observation.
        Observation structure: [grid (200), metadata (6), deck info, hold info]
        Grid is 10x20, we need to find the highest occupied cell.
        """
        # Extract grid (first 200 elements for 10x20 board)
        grid = obs[:200].reshape(20, 10)

        # Find highest occupied row (from top)
        for row in range(20):
            if np.any(grid[row] > 0):
                return 20 - row
        return 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get current board height
        current_height = self._get_board_height(obs)

        # Penalize height increase, reward height decrease
        height_change = current_height - self.last_height
        reward -= height_change * self.height_penalty_scale

        self.last_height = current_height

        return obs, reward, terminated, truncated, info


class RotationCurriculumWrapper(gym.Wrapper):
    """
    Curriculum learning: Force agent to rotate by masking non-rotation actions early in training.
    Gradually relax the constraint as training progresses.
    """

    def __init__(self, env, initial_rotation_prob=0.3, decay_steps=100000):
        super().__init__(env)
        self.initial_rotation_prob = initial_rotation_prob
        self.decay_steps = decay_steps
        self.current_step = 0

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        # With probability p, force a rotation instead of the chosen action
        # Probability decays over time
        rotation_prob = self.initial_rotation_prob * max(0, 1 - self.current_step / self.decay_steps)

        if np.random.random() < rotation_prob and action != 3:  # 3 = ACTION_ROTATE
            action = 3  # Force rotation

        self.current_step += 1

        return self.env.step(action)
