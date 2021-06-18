# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
import gym

from typing import Tuple
from gym_minigrid.wrappers import ViewSizeWrapper
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

def create_env(flags):
    env = gym.make(flags.env)
    env = ViewSizeWrapper(env, agent_view_size=flags.vision)
    env = Minigrid2Image(env)
    return env

def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment (CartPole-v0)
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
    output_dim = env.action_space.n
    return input_dim, output_dim

# Helper functions and wrappers
def _format_observation(obs):
    obs = torch.tensor(obs)
    return obs.view((1, 1) + obs.shape)  # (...) -> (T,B,...).


class Minigrid2Image(gym.ObservationWrapper):
    """Get MiniGrid observation to ignore language instruction."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, observation):
        return observation['image']

class Observation_WrapperSetup:
    """Environment wrapper to format observation items into torch."""
    def __init__(self, gym_env, fix_seed=False, env_seed=1):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.episode_win = None
        self.fix_seed = fix_seed
        self.env_seed = env_seed

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        if self.fix_seed:
            self.gym_env.seed(seed=self.env_seed)
        initial_frame = _format_observation(self.gym_env.reset())


        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            episode_win=self.episode_win,
            )

    def step(self, action):
        frame, reward, done, _ = self.gym_env.step(action.item())

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        episode_return = self.episode_return

        if done and reward > 0:
            self.episode_win[0][0] = 1
        else:
            self.episode_win[0][0] = 0
        episode_win = self.episode_win

        if done:
            if self.fix_seed:
                self.gym_env.seed(seed=self.env_seed)
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
            self.episode_win = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_observation(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)


        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step = episode_step,
            episode_win = episode_win,
            )

    def get_full_obs(self):
        env = self.gym_env.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        return full_grid

    def close(self):
        self.gym_env.close()

