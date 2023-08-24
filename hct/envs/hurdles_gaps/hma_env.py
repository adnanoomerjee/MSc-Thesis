# pylint:disable=g-multiple-import
"""Creates an environment for the lowest level of a hierarchical framework"""
import sys
import inspect
import os

from hct.envs.hurdles_gaps.hurdlesgaps_attributes import *

from hct.envs.goal import Goal
from hct.envs.tools import *
from hct.envs.ant_test import AntTest
from hct.training.configs import NetworkArchitecture, SMALL_TRANSFORMER_CONFIGS, DEFAULT_MLP_CONFIGS
from hct.io import model
from hct.envs.hma_env import HMA2Env, HMA3Env

from brax import base, generalized
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward

from etils import epath

import jax
from jax import numpy as jp

from typing import Optional, Literal, Tuple

from absl import logging


class HMA3HurdlesGapsEnv(HMA3Env):

  def __init__(
    self,
    task: Literal['ant_hurdles', 'ant_gaps'],
    mid_level_modelpath: str, 
    task_information: bool = False,
    reward_type: Literal['sparse', 'dense'] = 'sparse',
    reward_movement: Literal['position', 'velocity'] = 'velocity',
    action_repeat=1,
    architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
    **kwargs
  ):
    
    super().__init__(
      task=task,
      mid_level_modelpath=mid_level_modelpath,
      action_repeat=action_repeat,
      architecture_configs=architecture_configs,
      **kwargs)

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')
    self.task = task
    self.task_information = task_information

    setup(self)

    # Reward attributes
    self.reward_movement = reward_movement
    self.reward_type = reward_type 

    self.episode_length = 1500/self.mid_level_env.action_repeat


  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    return reset(self, rng)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    return step(self, state, action)

  def _reward(self, obstacles_complete, is_unhealthy,  state: base.State):
    return reward(self, obstacles_complete, is_unhealthy, state)
  


class HMA2HurdlesGapsEnv(HMA2Env):

  def __init__(
    self,
    task: Literal['ant_hurdles', 'ant_gaps'],
    low_level_modelpath: str, 
    task_information: bool = False,
    reward_type: Literal['sparse', 'dense'] = 'sparse',
    reward_movement: Literal['position', 'velocity'] = 'velocity',
    action_repeat=1,
    architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
    **kwargs
  ):
    
    #jax.config.update("jax_debug_nans", True)

    super().__init__(
      task=task,
      low_level_modelpath=low_level_modelpath,
      action_repeat=action_repeat,
      architecture_configs=architecture_configs,
      **kwargs)

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')
    self.task = task
    self.task_information = task_information

    setup(self)

    # Reward attributes
    self.reward_movement = reward_movement
    self.reward_type = reward_type 

    self.episode_length = 1500/self.low_level_env.action_repeat

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    return reset(self, rng)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    return step(self, state, action)

  def _reward(self, obstacles_complete, is_unhealthy,  state: base.State):
    return reward(self, obstacles_complete, is_unhealthy, state)
  


      


'''
def _sample_goal(self, rng: jp.ndarray, state: base.State):
    """
    Samples normalised goal and outputs a goal state 
    
    Goal is restricted to ensure a valid state, and acheive a range of positions
    that are expected to be achieved by the optimal policy. Restictions on
      Z position of all links
      Root rotation
      Number of feet in contact with ground:
        Randomly sample an ordered subset of end effector IDs
        Ensure that these end effectors are in contact with ground

    Args:
      rng: jp.ndarray

    Returns:
      goal: Goal
    """

    rng, rng1, rng2 = jax.random.split(rng, 3)
    randint = jax.random.randint(rng1,shape=(1,), minval = 0, maxval = 4)
    randchoice = jax.random.choice(rng2, jp.array([self.goal_contact_cond, self.goal_z_cond[1]]), shape=(1,), p = jp.array([0.9, 0.1]))
    goal_contact_cond = jp.full((4,), self.goal_z_cond[1]).at[randint].set(randchoice)

    def _create_goal(g: jp.ndarray, state: base.State) -> Goal:
      """
      Creates a goal state 

      Args:
        g: jp.ndarray normalised goal
        state: environment state

      Returns:
        Goal: goal state
      """
      q = q_spherical_to_quaternion(g[:self.dof], state, self.sys)
      qd = g[-self.dof:]
      x_world, xd_world = forward(self.sys, q, qd)
      x_world, xd_world = x_world * self.goal_x_mask, xd_world * self.goal_xd_mask
      x_rel, xd_rel = world_to_relative(x_world, self.sys), world_to_relative(xd_world, self.sys)
      x_rel, xd_rel = x_rel * self.goal_x_mask, xd_rel * self.goal_xd_mask
      return Goal(g, q, qd, x_world, x_rel, xd_world, xd_rel)

    def _sample(carry):
      rng, _ = carry
      rng, rng1 = jax.random.split(rng)
      g = jax.random.uniform(
        rng1, 
        shape=self.goal_size, 
        minval=self.goalsampler_limit[:, 0], 
        maxval=self.goalsampler_limit[:, 1]
      )
      goal = _create_goal(g, state)
      return rng, goal
    
    def _reject(carry):
      rng, goal = carry
      #cpos_foot_z = contact(self.sys, goal.x).pos[foot_contact_idx, 2]
      foot_z = goal.x_world.pos[self.end_effector_idx, 2]
      z = goal.x_world.pos[:,2]
      polar = goal.g[4]
      cond = \
        jp.any(z < self.goal_z_cond[0]) | jp.any(z > self.goal_z_cond[1]) | \
        (polar > self.goal_polar_cond) | \
        jp.any(foot_z > goal_contact_cond)
      return cond
    
    init_g = jax.random.uniform(
      rng, 
      shape=self.goal_size, 
      minval=self.goalsampler_limit[:, 0], 
      maxval=self.goalsampler_limit[:, 1]
    )

    init_val = rng, _create_goal(init_g, state)
    goal = jax.lax.while_loop(_reject, _sample, init_val)[1]

    return goal

  def _sample_goal(self, rng: jp.ndarray, state: base.State):
    """
    Samples normalised goal and outputs a goal state 
    
    Goal is restricted to ensure a valid state, and acheive a range of positions
    that are expected to be achieved by the optimal policy. Restictions on
      Z position of all links
      Root rotation
      Number of feet in contact with ground:
        Randomly sample an ordered subset of end effector IDs
        Ensure that these end effectors are in contact with ground

    Args:
      rng: jp.ndarray

    Returns:
      goal: Goal
    """

    def _create_goal(g: jp.ndarray, state: base.State) -> Goal:
      """
      Creates a goal state 

      Args:
        g: jp.ndarray normalised goal
        state: environment state

      Returns:
        Goal: goal state
      """
      
      q = q_spherical_to_quaternion(g[:self.dof], state, self.sys)
      qd = g[-self.dof:]
      x_world, xd_world = forward(self.sys, q, qd)
      x_world, xd_world = x_world * self.goal_x_mask, xd_world * self.goal_xd_mask
      x_rel, xd_rel = world_to_relative(x_world, self.sys), world_to_relative(xd_world, self.sys)
      x_rel, xd_rel = x_rel * self.goal_x_mask, xd_rel * self.goal_xd_mask
      return Goal(g, q, qd, x_world, x_rel, xd_world, xd_rel)

    rng, rng1, rng2 = jax.random.split(rng, 3)
    g = jax.random.uniform(
      rng, 
      shape=self.goal_size, 
      minval=self.goalsampler_limit[:, 0], 
      maxval=self.goalsampler_limit[:, 1]
    )
    goal = _create_goal(g, state)
    z = goal.x_world.pos[:,2]
    z_root = z[0] - jp.min(z) + self.goal_z_cond[0] 
    z_root = z_root + jax.random.choice(rng1, jp.array([0, 1]), p=jp.array([0.9, 0.1])) * jax.random.uniform(rng2, minval=z_root, maxval=self.goal_z_cond[1]-z_root)

    q = goal.q.at[2].set(z_root)
    qd = goal.qd
    x_world, xd_world = forward(self.sys, q, qd)
    x_world, xd_world = x_world * self.goal_x_mask, xd_world * self.goal_xd_mask
    x_rel, xd_rel = world_to_relative(x_world, self.sys), world_to_relative(xd_world, self.sys)
    x_rel, xd_rel = x_rel * self.goal_x_mask, xd_rel * self.goal_xd_mask   

    return Goal(g, q, qd, x_world, x_rel, xd_world, xd_rel)

    
'''