
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

y_sizes = [0.24, 0.23, 0.22, 0.34, 0.24, 0.34, 0.23, 0.24, 0.39, 0.38, 0.29, 0.39, 0.39, 0.36, 0.3, 0.36, 0.34, 0.3, 0.36, 0.31]
hurdle_sizes = [0.17, 0.12, 0.23, 0.2, 0.25, 0.23, 0.17, 0.24, 0.1, 0.28, 0.2, 0.28, 0.27, 0.23, 0.26, 0.23, 0.16, 0.11, 0.26, 0.16]

obstacle_completion_coords = [2.48, 4.74, 7.08, 9.2, 11.68, 14.44, 16.54, 18.4, 20.76, 23.58, 26.26, 28.44, 31.26, 33.28, 35.92, 38.56, 40.56, 43.06, 45.42, 47.96]

def setup(env):
  if env.task == 'ant_hurdles':
      completion_coords = obstacle_completion_coords[0::2]
      widths = jp.array(y_sizes[0::2]).reshape(-1, 1)
      heights = jp.array(hurdle_sizes[0::2]).reshape(-1, 1)
      obstacle_information = jp.concatenate([widths, heights], axis=-1)
  else:
      completion_coords = obstacle_completion_coords
      widths = jp.array(y_sizes).reshape(-1, 1)
      obstacle_information = widths

  env.obstacle_information = obstacle_information
  env.obstacle_completion_coords = jp.array(completion_coords)
  env.num_obstacles = len(completion_coords)

def reward(env, obstacles_complete, is_unhealthy, state: base.State):
    
    next_obstacle_coord = env.obstacle_completion_coords[jp.array(obstacles_complete).astype(int)]
    passed_next_obstacle = state.x.pos[0, 1] >= next_obstacle_coord

    def sparse_reward():
      return 1.0 * passed_next_obstacle

    def dense_reward():
        return state.xd.vel[0, 1]
    
    if env.reward_type == 'dense':
      reward = dense_reward()
    else:
      reward = sparse_reward()

    obstacles_complete = passed_next_obstacle * (obstacles_complete + 1) + (1 - passed_next_obstacle) * obstacles_complete
    return reward, obstacles_complete


def reset(env, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    q = env.sys.init_q 
    qd = jp.zeros((env.sys.qd_size(),))

    pipeline_state = env.pipeline_init(q, qd)

    # Get observation
    obs = get_obs(env, pipeline_state, jp.array(0.0))    

    # Set metrics
    done = 0.0

    reward, _ = env._reward(0.0, 0.0, pipeline_state)

    metrics = {
      'reward': reward,
      'obstacles_complete': 0.0,
      'is_unhealthy': 0.0,
      'task_complete': 0.0
    }

    info = {
      'rng': rng,
      'goal': None,
      'first_pipeline_state': pipeline_state,
      'first_obs': obs
    }

    return State(pipeline_state, obs, reward, done, metrics, info)


def step(env, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    if action.shape[-1] == 1:
      action = jp.squeeze(action, axis=-1) 

    prev_done = state.done
    obstacles_complete = state.metrics['obstacles_complete'] * 1.0 * (1 - (jp.all(state.pipeline_state.x.pos[0, :2] == 0.0)))

    rng = state.info['rng']

    # Take action
    prev_pipeline_state = state.pipeline_state

    pipeline_state, rng = env.take_action(prev_pipeline_state, action, rng)

    # Check if unhealthy
    is_unhealthy = jp.where(rotate(jp.array([0, 0, 1]), pipeline_state.x.rot[0])[-1] < 0, x=1.0, y=0.0)
    is_unhealthy = jp.where(jp.all(pipeline_state.x.pos[:, 2] > -0.5), x=is_unhealthy, y=1.0)

    reward, obstacles_complete = env._reward(obstacles_complete, is_unhealthy, pipeline_state)

    # Check if goal reached
    goal_reached = obstacles_complete >= env.num_obstacles

    # Get observation
    obs = get_obs(env, pipeline_state, obstacles_complete)

    pipeline_state = jax.tree_map(lambda w, z: jp.where(prev_done, x=w, y=z), prev_pipeline_state, pipeline_state)
    obs = jax.tree_map(lambda w, z: jp.where(prev_done, x=w, y=z), state.obs, obs)

    done = 0.0 + jp.logical_or(is_unhealthy, goal_reached)

    reward = (reward * (1-done) - done * is_unhealthy) * (1-prev_done) 

    state.metrics.update(
      reward=reward,
      obstacles_complete=obstacles_complete,
      is_unhealthy=1.0*is_unhealthy,
      task_complete=1.0*goal_reached
    )

    state.info.update(
       rng=rng
    )
    
    return state.replace(
      pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

def get_obs(env, pipeline_state, obstacles_complete):
    
    obs = env.get_obs(pipeline_state)

    jp.where(obstacles_complete == env.num_obstacles, x=0.0, y=obstacles_complete)

    if env.task_information:
       o = (obstacles_complete).astype(int)
       distance_to_next_obstacle = jp.expand_dims(env.obstacle_completion_coords[o] - pipeline_state.x.pos[0, 1], axis=0)
       obstacle_information = env.obstacle_information[o]
       if env.network_architecture.name == 'MLP':
          obs = (jp.concatenate([obs, distance_to_next_obstacle, obstacle_information])).squeeze()
       else:
          d = jp.expand_dims(jp.repeat(distance_to_next_obstacle, env.num_nodes), -1)
          i = jp.repeat(obstacle_information, env.num_nodes)
          obs = jp.concatenate([obs, d, i], axis=-1)
    
    return obs