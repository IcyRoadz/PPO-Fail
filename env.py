import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleSim(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SimpleSim, self).__init__()
        self.num_actions = 10
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.MultiBinary(self.num_actions)
        self.selection_reward = 0.1
        self.repeated_selection_penalty = 0
        self.all_selected_reward = 0
        self.episode_end_penalty = 0

    def reset(self, *, seed=None, options=None):
        self.state = np.zeros(self.num_actions, dtype=int)  # Reset action
        return self.state, {}

    def step(self, action):
        done = False
        reward = 0

        if self.state[action] == 0:
            # Package has not been selected before
            self.state[action] = 1
            reward = self.selection_reward
        else:
            # Package was selected before
            reward = self.repeated_selection_penalty
            done = True

        if np.all(self.state == 1):
            # All packages have been selected
            done = True

        return self.state, reward, done, False, {}
