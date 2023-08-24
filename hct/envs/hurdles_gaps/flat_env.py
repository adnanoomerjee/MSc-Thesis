# pylint:disable=g-multiple-import
"""Creates an environment for the lowest level of a hierarchical framework"""
import sys
import inspect
import os

from hct.envs.goal import Goal
from hct.envs.tools import *
from hct.envs.ant_test import AntTest
from hct.training.configs import NetworkArchitecture, SMALL_TRANSFORMER_CONFIGS, DEFAULT_MLP_CONFIGS
from hct.io import model
from hct.envs.hurdles_gaps.hurdlesgaps_attributes import *

from brax import base, generalized
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward

from etils import epath

import jax
from jax import numpy as jp

from typing import Optional, Literal, Tuple

from absl import logging


class FlatHurdlesGapsEnv(PipelineEnv):

  def __init__(
    self,
    task = Literal['ant_hurdles', 'ant_gaps'],
    task_information: bool = False,
    reward_type: Literal['sparse', 'dense'] = 'sparse',
    reward_movement: Literal['position', 'velocity'] = 'velocity',
    architecture_name='MLP',
    architecture_configs=DEFAULT_MLP_CONFIGS, # trial larger network
    **kwargs
  ):

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')

    logging.info('Initialising environment...')
  
    path = epath.resource_path('hct') / f'envs/assets/{task}.xml'
    sys = mjcf.load(path)

    n_frames = 5

    backend = 'positional'

    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.005)
      n_frames = 10

    if backend == 'positional':
      sys = sys.replace(
        actuator=sys.actuator.replace(
          gear=200 * jp.ones_like(sys.actuator.gear)
        )
      )

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self.task = task

    setup(self)

    # Agent attributes
    self.dof = jp.array(self.sys.dof.limit).T.shape[0]
    self.num_links = sys.num_links()
    self.link_parents = sys.link_parents
    self.task_information = task_information

    # Reward attributes
    self.reward_movement = reward_movement
    self.reward_type = reward_type 

    self.non_actuator_nodes = 0 
    self.action_mask = None
    self.num_nodes = 9
    self.obs_mask = None

    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name=architecture_name, **architecture_configs)
    num_attn_heads = self.network_architecture.configs['policy_num_heads'] if self.network_architecture.name=='Transformer' else 2
    self.max_actions_per_node = 1 if self.network_architecture.name=='Transformer' else 8

    # Observation attributes
    self.root_mask = base.Transform(jp.array([0.0, 0.0, 1.0]), jp.array([1.0, 1.0, 1.0, 1.0]))
    self.state_obs_width = 13
    concat_obs_width = self.state_obs_width 
    if concat_obs_width % num_attn_heads != 0:
        self.concat_obs_width = ((concat_obs_width // num_attn_heads) + 1) * num_attn_heads
    else:
        self.concat_obs_width = concat_obs_width

    self.action_repeat = 1
    self.episode_length = 1500
    self.action_shape = (8, 1)

    logging.info('Environment initialised.')

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    return reset(self, rng)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    return step(self, state, action)
  
  def take_action(self, state, action, rng):
    return self.pipeline_step(state, action), rng
    
  def get_obs(self, state: base.State) -> jp.ndarray:
    """
    Processes a state and goal into observation format

    Args:
        state: dynamic State object, changing every step
        goal: Goal object, containing egocentric Transform goals
          and/or world frame velocity goals

    Returns:
        obs: (num_links, 13 + goal_size) array containing goal observations:
    """
    
    def _get_state_obs(state: Union[base.State, Goal]):
      """Returns world root position (masking XY position) and limb position relative to parent"""
      return self._world_to_relative(state)
 
    sx, sxd =  _get_state_obs(state)
    sx, sxd = concatenate_attrs(sx), concatenate_attrs(sxd)
    obs = jp.concatenate([sx, sxd], axis = -1)
    if self.network_architecture.name == 'MLP':
      obs = obs.reshape(*obs.shape[:-2], -1)
    return obs
  
  def _reward(self, obstacles_complete, is_unhealthy,  state: base.State):
    return reward(self, obstacles_complete, is_unhealthy, state)
  
  def _world_to_relative(self, state: base.State):
    return world_to_relative(state, self)
  