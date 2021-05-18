"""Agent for a Walker2DBullet environment."""
# from os.path import abspath, dirname, realpath

from gym.spaces.box import Box
import numpy as np

import sys, os
sys.path.append(os.pardir)
from models import *
from ee619.TD3 import *

# ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self):
        self._action_space = Box(-1, 1, (6,))
        self._action_space.seed(0)

        kwargs = {
            "state_dim": 22,
            "action_dim": 6,
            "max_action": 1.0,
            "discount": 0.99,
            "tau": 0.005,
            "policy_noise" : 0.2,
            "noise_clip" : 0.5,
            "policy_freq" : 2,
        }

        self.policy = TD3(**kwargs)

    def act(self, observation: np.ndarray):
        """Decides which action to take for the given observation."""
        # observation : numpy.ndarray
        self._action_space = self.policy.select_action(np.array(observation))
        del observation
        return self._action_space

    def load(self):
        self.policy.load(f"./models/TD3_Walker2DBulletEnv-v0")
        

    