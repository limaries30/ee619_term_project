"""Agent for a Walker2DBullet environment."""
from os.path import abspath, dirname, realpath

from gym.spaces.box import Box
import numpy as np
from .model import GaussianPolicy, QNetwork, DeterministicPolicy
from .utils import soft_update, hard_update, yaml_to_class

import yaml
import os
from os.path import abspath, dirname, realpath

import torch
import torch.nn.functional as F
from torch.optim import Adam


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory

YAML_PATH = './conf.yaml'

if os.path.isfile(YAML_PATH):
    with open(YAML_PATH) as f:
        args = yaml_to_class(yaml.safe_load(f))
else:
    print('no yaml file')



class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self):

        self._action_space = Box(-1, 1, (6,))
        self._action_space.seed(0)
        self._state_space = 22

        args.action_space = self._action_space
        args.num_inputs = self._state_space

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(args.num_inputs, args.action_space.shape[0], args.hidden_size,args.isDropout).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(args.num_inputs, args.action_space.shape[0], args.hidden_size,args.isDropout).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(args.action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(args.num_inputs, args.action_space.shape[0], args.hidden_size, args.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(args.num_inputs, args.action_space.shape[0], args.hidden_size, args.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def act(self, observation: np.ndarray,evaluate=False):
        """Decides which action to take for the given observation."""
        state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, batch_size,
                          next_q_value, updates):

        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


    def load(self):
        print(self.actor_path)
        print(self.critic_path)
        try:
            if self.actor_path is not None:
                self.policy.load_state_dict(torch.load(self.actor_path))
            if self.critic_path is not None:
                self.critic.load_state_dict(torch.load(self.critic_path))
        except:
            raise ValueError
            pass