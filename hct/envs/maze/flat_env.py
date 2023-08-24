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
from hct.envs.maze.maze_attributes import *

from brax import base, generalized
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward

from etils import epath

import jax
from jax import numpy as jp

from typing import Optional, Literal, Tuple

from absl import logging


class FlatMazeEnv(PipelineEnv):

  def __init__(
    self,
    task_information: bool = False,
    reward_per_milestone = 1,
    reward_sparsity = 1,
    architecture_name='MLP',
    architecture_configs=DEFAULT_MLP_CONFIGS, # trial larger network
    reward_type: Literal['sparse', 'sparse_with_information', 'dense'] = 'sparse',
    reward_movement: Literal['position', 'velocity'] = 'velocity',
    **kwargs
  ):

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')

    logging.info('Initialising environment...')
  
    path = epath.resource_path('hct') / f'envs/assets/ant_maze.xml'
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

    # Agent attributes
    self.dof = jp.array(self.sys.dof.limit).T.shape[0]
    self.num_links = sys.num_links()
    self.link_parents = sys.link_parents
    self.task_information = task_information

    # Reward attributes
    self.reward_milestone = reward_sparsity * reward_per_milestone
    self.reward_sparsity = reward_sparsity
    self.reward_type = reward_type
    self.reward_movement = reward_movement
    
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
    self.episode_length = 1000
    self.action_shape = (8, 1)

    logging.info('Environment initialised.')

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    return reset(self, rng)
    
  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    return step(self, state, action)
  
  def take_action(self, state: State, action: jp.array, rng):
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

  def _world_to_relative(self, state: base.State):
    return world_to_relative(state, self)
  


      


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