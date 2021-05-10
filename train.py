"""Evaluates an agent on a Walker2DBullet environment."""
from argparse import ArgumentParser, Namespace
from math import fsum
from pickle import dump
from typing import List, Optional

import gym
from gym import logger
# pybullet_envs must be imported in order to create Walker2DBulletEnv
import pybullet_envs    # noqa: F401  # pylint: disable=unused-import

from ee619.agent import Agent


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
    logger.set_level(logger.DISABLED)
    env = gym.make('Walker2DBulletEnv-v0')
    agent.load()
    rewards: List[float] = []

    num_actions = env.action_space
    num_states = env.observation_space

    #print(num_states,num_actions)

    for seed_ in range(seed, seed + num_episodes):
        env.seed(seed_)
        observation = env.reset()

        #print('initial observation',observation.shape)
        done = False
        episode_steps = 0
        while not done:
            env.render()
            action = agent.act(observation)

            observation, reward, done, _ = env.step(action)

            rewards.append(reward)

            episode_steps += 1
            #print("episode_steps",episode_steps)

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
