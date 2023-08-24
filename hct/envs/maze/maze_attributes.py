import sys
import inspect
import os

from hct.envs.goal import Goal
from hct.envs.tools import *
from hct.envs.ant_test import AntTest
from hct.training.configs import NetworkArchitecture, SMALL_TRANSFORMER_CONFIGS, DEFAULT_MLP_CONFIGS
from hct.io import model

from brax import base, generalized
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward

from etils import epath

import jax
from jax import numpy as jp

from typing import Optional, Literal, Tuple

from absl import logging

reward_per_milestone = 100
reward_sparsity = 1

maze_cell_values = -jp.array([
      [0., 19., 18., 17., 18.],
      [1.,  2., 15., 16., 19.],
      [4.,  3., 14., 13., 20.],
      [5.,  8.,  9., 12., 21.],
      [6.,  7., 10., 11., 22.],
    ])

next_maze_cell = jp.array([
    [[0, 0], [0, 2], [0, 3], [1, 3], [0, 3]],
    [[0, 0], [1, 0], [2, 2], [1, 2], [0, 4]],
    [[2, 1], [1, 1], [2, 3], [3, 3], [1, 4]],
    [[2, 0], [4, 1], [3, 1], [4, 3], [2, 4]],
    [[3, 0], [4, 0], [3, 2], [4, 2], [3, 4]],
], dtype=jp.int32)

cell_centroid = jp.array([
    [[-8,  8], [-4,  8], [0,  8], [4,  8], [8,  8]],
    [[-8,  4], [-4,  4], [0,  4], [4,  4], [8,  4]],
    [[-8,  0], [-4,  0], [0,  0], [4,  0], [8,  0]],
    [[-8, -4], [-4, -4], [0, -4], [4, -4], [8, -4]],
    [[-8, -8], [-4, -8], [0, -8], [4, -8], [8, -8]],
], dtype=jp.int32)


def get_maze_coordinates(state: base.State):
    xy = jp.mean(state.x.pos[:, :2], axis=0)
    xpos, ypos = xy
    column = (1 + (xpos + 10) // 4 + (abs(xpos) == 2))
    row = (5 - (ypos + 10) // 4 + (abs(ypos) == 2))
    coords =  (row, column)
    value = maze_cell_values[row.astype(int)-1, column.astype(int)-1]
    next_cell  = next_maze_cell[row.astype(int)-1, column.astype(int)-1, :]
    return coords, value, xy, next_cell

def get_obs(env, pipeline_state, next_cell_centroid):
    obs = env.get_obs(pipeline_state)
    if env.task_information:
      distance_to_next_cell = next_cell_centroid - pipeline_state.x.pos[0, :2] 
      if env.network_architecture.name == 'MLP':
        obs = (jp.concatenate([obs, distance_to_next_cell])).squeeze()
      else:
        d = jp.expand_dims(jp.repeat(distance_to_next_cell, env.num_nodes), -1)
        obs = jp.concatenate([obs, d], axis=-1)
    return obs

def reset(env, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    q = env.sys.init_q 
    qd = qd = jp.zeros((env.sys.qd_size(),))

    pipeline_state = env.pipeline_init(q, qd)

    maze_coord, state_value, xy_pos, next_cell = get_maze_coordinates(pipeline_state)

    next_cell_centroid = cell_centroid[*next_cell]

    # Get observation
    obs = get_obs(env, pipeline_state, next_cell_centroid)    

    # Set metrics
    done, reward = jp.array([0.0, -1.0])

    if env.reward_type == 'dense':
      reward = -22.0 if env.reward_movement == 'position' else 0.0

    metrics = {
      'reward': reward,
      'maze_row': maze_coord[0],
      'maze_column': maze_coord[1],
      'distance_to_end': -state_value,
      'is_unhealthy': 0.0,
      'task_complete': 0.0
    }

    info = {
      'rng': rng,
      'goal': None,
      'prev_reward_distance': 22.0,
      'first_pipeline_state': pipeline_state,
      'first_obs': obs
    }

    return State(pipeline_state, obs, reward, done, metrics, info)


def step(env, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    if action.shape[-1] == 1:
      action = jp.squeeze(action, axis=-1) 

    rng = state.info['rng']

    pipeline_state, rng = env.take_action(state.pipeline_state, action, rng)

    # Check if unhealthy
    is_unhealthy = jp.where(rotate(jp.array([0, 0, 1]), pipeline_state.x.rot[0])[-1] < 0, x=1.0, y=0.0)

    maze_coord, state_value, xy_pos, next_cell = get_maze_coordinates(pipeline_state)

    prev_reward_distance = state.info['prev_reward_distance']

    distance_to_end = -state_value

    next_cell_centroid = cell_centroid[*next_cell]

    # Get observation
    obs = get_obs(env, pipeline_state, next_cell_centroid)

    # Check if goal reached
    goal_reached = state_value == 0
    milestone_reached =  prev_reward_distance - distance_to_end >= env.reward_sparsity

    if env.reward_type == 'sparse':
      reward = env.reward_milestone * (milestone_reached + goal_reached) 
      reward = reward * (1 - is_unhealthy) - is_unhealthy
    else:
      if env.reward_movement == 'position':
        next_cell_value = maze_cell_values[*next_cell]
        reward = next_cell_value - safe_norm(xy_pos - next_cell_centroid)
      else:
        reward = jp.dot(pipeline_state.xd.vel[0, :2], normalize(next_cell_centroid - xy_pos)[0])

    reward = reward * (1-state.done)

    done = 0.0 + jp.logical_or(is_unhealthy, goal_reached)

    prev_reward_distance = (distance_to_end * milestone_reached + prev_reward_distance * (1-milestone_reached)) * (1 - done) + done * 22.0

    state.info.update(
      prev_reward_distance=prev_reward_distance,
      rng=rng
    )
    
    state.metrics.update(
      reward=reward,
      maze_row=maze_coord[0],
      maze_column=maze_coord[1],
      distance_to_end=distance_to_end,
      is_unhealthy=1.0*is_unhealthy,
      task_complete=1.0*goal_reached
    )
    return state.replace(
      pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
