"""Agent for a Walker2DBullet environment."""
from os.path import abspath, dirname, realpath

from gym.spaces.box import Box
import numpy as np

import yaml
import os


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory

YAML_PATH = './ee619/conf.yaml'
if os.path.isfile(YAML_PATH):
    with open(YAML_PATH) as f:
        conf = yaml.safe_load(f)
else:
    print('no yaml file')

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self):
        self._action_space = Box(-1, 1, (6,))
        self._action_space.seed(0)

        self._state_space = 22

    def act(self, observation: np.ndarray):
        """Decides which action to take for the given observation."""
        del observation
        return self._action_space.sample()

    def load(self):
        """Loads network parameters if there are any.

        Example:
            path = join(ROOT, 'model.pth')
            self.policy.load_state_dict(torch.load(path))
        """

