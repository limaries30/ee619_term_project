"""Evaluates an agent on a Walker2DBullet environment."""
from argparse import ArgumentParser, Namespace
from math import fsum
from pickle import dump
from typing import List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import yaml
import os
import datetime
import itertools

import gym
from gym import logger
# pybullet_envs must be imported in order to create Walker2DBulletEnv
import pybullet_envs    # noqa: F401  # pylint: disable=unused-import

from ee619.agent import Agent
from ee619.replay_memory import ReplayMemory


def parse_args() -> Namespace:
    """Parses arguments for evaluate()"""
    parser = ArgumentParser(
        description='Evaluates an agent on a Walker2DBullet environment.')
    parser.add_argument('-l', dest='label', default=None,

                        help='if unspecified, the mean episodic return will be '
                             'printed to stdout. otherwise, it will be dumped '
                             'to a pickle file of the given path.')
    parser.add_argument('-num_episodes', type=int, dest='num_episodes', default=100,
                        help='number of trials.')
    parser.add_argument('-s', type=int, dest='seed', default=0,
                        help='passed to the environment for determinism.')
    return parser.parse_args()


def train(agent: Agent, label: Optional[str], num_episodes: int, seed: int):
    """Computes the mean episodic return of the agent.

    Args:
        agent: The agent to evaluate.
        label: If None, the mean episodic return will be printed to stdout.
            Otherwise, it will be dumped to a pickle file of the given name
            under the "data" directory.
        repeat: Number of trials.
        seed: Passed to the environment for determinism.
    """
    YAML_PATH = './conf.yaml'
    if os.path.isfile(YAML_PATH):
        with open(YAML_PATH) as f:
            conf = yaml.safe_load(f)
    else:
        print('no yaml file')

    print('conf',conf)

    logger.set_level(logger.DISABLED)
    env = gym.make('Walker2DBulletEnv-v0')
    env.seed(seed)

    #print('env._max_episode_steps',env._max_episode_steps) #1000

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent.load()

    # Tesnorboard
    writer = SummaryWriter(
        'runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                   conf['policy'], "autotune" if conf['automatic_entropy_tuning'] else ""))

    # Memory
    memory = ReplayMemory(conf['replay_size'], seed)



    rewards: List[float] = []

    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):

        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        episode_steps = 0
        while not done:

            if conf['start_steps'] > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.act(state)  # Sample action from policy

            if len(memory) > conf['batch_size']:
                # Number of updates per step in environment
                for i in range(conf['updates_per_step']):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         conf['batch_size'],
                                                                                                         updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

            rewards.append(reward)

        if total_numsteps > conf['num_steps']:
            break
        writer.add_scalar('reward/train', episode_reward, i_episode)
        if i_episode % 100 == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))

        if i_episode % 10 == 0 and conf['eval'] is True:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.act(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    mean_episode_return = fsum(rewards) / num_episodes
    if label is None:
        print(mean_episode_return)
    else:
        if not label.endswith('.pkl'):
            label += '.pkl'
        with open(label, 'wb') as file:
            dump(mean_episode_return, file)
    return mean_episode_return


if __name__ == '__main__':
    train(agent=Agent(), **vars(parse_args()))
