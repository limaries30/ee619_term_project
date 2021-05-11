"""Agent for a Walker2DBullet environment."""
from os.path import abspath, dirname, realpath

from gym.spaces.box import Box
import numpy as np
from model import GaussianPolicy, QNetwork, DeterministicPolicy

import yaml
import os
from os.path import abspath, dirname, realpath

import torch
import torch.nn.functional as F
from torch.optim import Adam


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory

YAML_PATH = './ee619/conf.yaml'
if os.path.isfile(YAML_PATH):
    with open(YAML_PATH) as f:
        conf = yaml.safe_load(f)
        print('conf', conf)
        print('cuda', torch.cuda.is_available())
else:
    print('no yaml file')

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self):
        self._action_space = Box(-1, 1, (6,))
        self._action_space.seed(0)
        self._state_space = 22

        self.gamma = conf['gamma']
        self.tau = conf['tau']
        self.alpha = conf['alpha']

        self.policy_type = conf['policy']
        self.target_update_interval = conf['target_update_interval']
        self.automatic_entropy_tuning = conf['automatic_entropy_tuning']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(self._state_space, self._action_space.shape[0], conf['hidden_size']).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=conf['lr'])


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

