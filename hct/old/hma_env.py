
import inspect
import os

from hct.envs.goal import Goal
from hct.envs.ant_test import AntTest
from hct.envs.tools import *
from hct.training.configs import NetworkArchitecture, SMALL_TRANSFORMER_CONFIGS, DEFAULT_MLP_CONFIGS
from hct.io.model import load, save

from brax import base, generalized
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward

from etils import epath

import jax
from jax import numpy as jp

from typing import Callable, Optional, Literal, Tuple

from absl import logging

from brax.training.networks import FeedForwardNetwork

import sys as pysys

class HMA3Env(PipelineEnv):

  def __init__(
      self,
      task,
      mid_level_modelpath: str,
      state: Literal['abstract', 'below', 'full'] = 'abstract',
      action_repeat=5,
      architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
      **kwargs
  ):
    
    mid_level_network = load(f"{mid_level_modelpath}/network")
    mid_level_params = load(f"{mid_level_modelpath}/model_params")
    mid_level_make_inference_fn = load(f"{mid_level_modelpath}/make_inference_fn")
    mid_level_env = load(f"{mid_level_modelpath}/env")

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')

    logging.info('Initialising environment...')

    path = epath.resource_path('hct') / f'envs/assets/{task}.xml'
    sys = mjcf.load(path)

    mid_level_env.sys = sys

    n_frames = 5

    backend = mid_level_env.backend

    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.005)
      n_frames = 10

    if backend == 'positional':
      # TODO: does the same actuator strength work as in spring
      sys = sys.replace(
          actuator=sys.actuator.replace(
              gear=200 * jp.ones_like(sys.actuator.gear)
          )
      )

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    # Low level attributes
    
    self.mid_level_env = mid_level_env
    self.mid_level_policy = mid_level_make_inference_fn(mid_level_network)(
        mid_level_params,
        train=False,
        obs_mask=mid_level_env.obs_mask,
        action_mask=mid_level_env.action_mask,
        non_actuator_nodes=mid_level_env.non_actuator_nodes,
        deterministic=True
    )
    self.low_level_env = mid_level_env.low_level_env
    self.low_level_policy = mid_level_env.low_level_policy

    # Agent attributes
    self.dof = mid_level_env.dof
    self.num_links = mid_level_env.num_links
    self.link_parents = (-1)

    # Training attributes
    self.obs_mask = None
    self.non_actuator_nodes = None
    self.num_nodes = 1

    # Goal attributes
    self.position_goals = mid_level_env.position_goals
    self.velocity_goals = mid_level_env.velocity_goals
    self.root_velocity_goals = mid_level_env.root_velocity_goals
    self.full_velocity_goals = mid_level_env.full_velocity_goals
    self.goal_importance = mid_level_env.goal_importance
    self.importance_scaling = 40
    self.state = state

    # Observation attributes
    self.state_obs_width = mid_level_env.state_obs_width

    if self.position_goals and self.full_velocity_goals:
      max_actions_per_node = 60
    elif self.position_goals and self.root_velocity_goals:
      max_actions_per_node = 36
    else:
      max_actions_per_node = 30

    self.goal_information_last_idx = max_actions_per_node
    self.max_actions_per_node = max_actions_per_node + self.goal_importance * 5
    self.action_mask = None 
    
    self.mid_level_goal_ranges = jp.concatenate(
        [
          jp.array(self.mid_level_env.pos_range),
          jp.squeeze(jp.array(self.mid_level_env.rot_range)),
          jp.array(self.mid_level_env.vel_range_range),
          jp.array(self.mid_level_env.ang_range)
        ],
        axis=-1
      )

    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name='MLP', **architecture_configs)
    self.action_repeat = action_repeat
    logging.info(mid_level_env.no_root_goals)

    if int(action_repeat) < int(self.mid_level_env.action_repeat):
      pysys.exit()

    logging.info('Environment initialised.')


  def take_action(self, cur_pipeline_state: base.State, mid_level_goal: jp.array, rng):

    rng, rng1 = jax.random.split(rng)

    mid_level_goal = self.get_mid_level_goal(mid_level_goal, cur_pipeline_state)
    mid_level_obs, _ = self.mid_level_env.get_obs(cur_pipeline_state, mid_level_goal, goal_dist=False) 
    low_level_goal, _ = self.mid_level_policy(mid_level_obs, rng1) 
    low_level_goal = self.mid_level_env.get_low_level_goal(low_level_goal, cur_pipeline_state) 

    def low_level_goal_repeat(carry, unused_nstate):
      pipeline_state, rng = carry
      rng, rng1 = jax.random.split(rng)
      low_level_obs, _ = self.low_level_env.get_obs(pipeline_state, low_level_goal)
      action, _ = self.low_level_policy(low_level_obs, rng1) 
      nstate = self.pipeline_step(pipeline_state, action) 
      return (nstate, rng), nstate
    
    carry_init = cur_pipeline_state, rng
    (pipeline_state, rng), _ = jax.lax.scan(
      low_level_goal_repeat, carry_init, (), length=self.mid_level_env.action_repeat)
    
    return pipeline_state, rng
  

  def get_mid_level_goal(self, g, state):
    logging.info(self.goal_importance)

    if self.goal_importance:
      importance = g[self.goal_information_last_idx:]
      importance = importance.reshape(5, 1)
      if self.mid_level_env.goal_importance_framework == 'continuous':
        importance = (importance + 1)/2
      else:
        importance = jp.where(importance > 0, x=1.0, y=0.0)
    else:
      importance = None

    g = g[:self.goal_information_last_idx]

    if (self.position_goals and self.velocity_goals) or self.root_velocity_goals:
      root_goals = jp.expand_dims(g[:12], axis=0)
      limb_goals = g[12:].reshape(4, g[6:].shape[-1]//4)
      goals = jp.concatenate([root_goals, limb_goals], axis=-2)
    elif self.root_velocity_goals:
      root_goals = jp.expand_dims(g[:12], axis=0)
      limb_goals = g[12:].reshape(4, g[6:].shape[-1]//4)
      limb_goals = pad(limb_goals, root_goals.shape[-1])
      goals = jp.concatenate([root_goals, limb_goals], axis=-2)
    else:
      root_goals = jp.expand_dims(g[:6], axis=0)
      limb_goals = g[6:].reshape(4, g[6:].shape[-1]//4)
      goals = jp.concatenate([root_goals, limb_goals], axis=-2)

    if self.position_goals and not self.velocity_goals:
      goals = pad(goals, 12)
    elif self.velocity_goals_goals and not self.position_goals:
      goals = jp.concatenate([jp.zeros((5,6)) + goals], axis=-1)

    goals_unnormalised = unnormalize_to_range(
      goals,
      self.mid_level_goal_ranges[0],
      self.mid_level_goal_ranges[1],
      -1,
      1
    )

    pos_goals = goals_unnormalised[:, 0:3]
    pos_goals = pos_goals.at[0].set(goals_unnormalised[0, 0:3] + state.x.pos[0])
    rot_goals = goals_unnormalised[:, 3:6]
    vel_goals = goals_unnormalised[:, 6:9]
    ang_goals = goals_unnormalised[:, 9:12]

    x_rel = base.Transform(pos_goals, rot_goals)
    xd_rel = base.Motion(vel_goals, ang_goals)

    return Goal(None, None, None, None, x_rel, xd_rel, importance=importance)
  

  def get_obs(self, state: base.State) -> jp.ndarray:
    """Return observation input tensor"""
    if self.mid_level_env.state_below:
      state = world_to_egocentric(state)
      state = self.mid_level_env.abstract_state(state)
      sx = state.x
      sxd = state.xd
      obs = jp.concatenate([concatenate_attrs(sx), concatenate_attrs(sxd)], axis=-1)
      obs = obs.reshape(*obs.shape[:-2], -1)
    else:
      spos = jp.mean(state.x.pos, axis=0)
      srot = state.x.rot[0]
      svel = jp.mean(state.xd.vel, axis=0)
      sang = state.xd.ang[0]
      obs = jp.concatenate([spos, srot, svel, sang], axis = -1)

    return obs
  


class HMA2Env(PipelineEnv):

  def __init__(
      self,
      task,
      low_level_modelpath: str,
      action_repeat=5,
      state_below = False,
      no_root_goals = False,
      architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
      **kwargs
  ):
    
    low_level_network = load(f"{low_level_modelpath}/network")
    low_level_params = load(f"{low_level_modelpath}/model_params")
    low_level_make_inference_fn = load(f"{low_level_modelpath}/make_inference_fn")
    low_level_env = load(f"{low_level_modelpath}/env")

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')

    logging.info('Initialising environment...')

    path = epath.resource_path('hct') / f'envs/assets/{task}.xml'
    sys = mjcf.load(path)

    low_level_env.sys = sys

    n_frames = 5

    backend = low_level_env.backend

    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.005)
      n_frames = 10

    if backend == 'positional':
      # TODO: does the same actuator strength work as in spring
      sys = sys.replace(
          actuator=sys.actuator.replace(
              gear=200 * jp.ones_like(sys.actuator.gear)
          )
      )

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    # Low level attributes
    
    self.low_level_env = low_level_env
    self.low_level_policy = low_level_make_inference_fn(low_level_network)(
        low_level_params,
        train=False,
        obs_mask=low_level_env.obs_mask,
        action_mask=low_level_env.action_mask,
        non_actuator_nodes=low_level_env.non_actuator_nodes,
        deterministic=True
    )

    # Agent attributes
    self.dof = low_level_env.dof
    self.num_links = low_level_env.num_links
    self.link_parents = (-1)

    # Training attributes
    self.obs_mask = None
    self.non_actuator_nodes = None
    self.num_nodes = 5

    # Goal attributes
    self.position_goals = low_level_env.position_goals
    self.velocity_goals = low_level_env.velocity_goals
    self.root_velocity_goals = low_level_env.root_velocity_goals
    self.full_velocity_goals = low_level_env.full_velocity_goals
    self.goal_importance = low_level_env.goal_importance
    self.goal_importance_framework = low_level_env.goal_importance_framework
    self.state_below = state_below
    self.no_root_goals = no_root_goals

    # Observation attributes
    self.state_obs_width = low_level_env.state_obs_width

    if self.position_goals and self.full_velocity_goals:
      max_actions_per_node = 24 + 2 * self.goal_importance
      self.action_mask = jp.ones((self.num_nodes, max_actions_per_node)).at[0,12:24].set(0)
    elif self.position_goals and self.root_velocity_goals:
      max_actions_per_node = 18 + 2 * self.goal_importance
      self.action_mask = jp.ones((self.num_nodes, max_actions_per_node)).at[1:,6:12].set(0).at[0,12:18].set(0)
    else:
      max_actions_per_node = 12 + 2 * self.goal_importance
      self.action_mask = jp.ones((self.num_nodes, max_actions_per_node)).at[0,6:12].set(0)
    
    self.low_level_goal_ranges = jp.concatenate(
        [
          jp.array(self.low_level_env.pos_range),
          jp.squeeze(jp.array(self.low_level_env.rot_range)),
          jp.array(self.low_level_env.vel_range_range),
          jp.array(self.low_level_env.ang_range)
        ],
        axis=-1
      )
    
    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name=low_level_env.network_architecture.name, **architecture_configs)

    if self.network_architecture.name=='Transformer':
      num_attn_heads = self.network_architecture.configs['policy_num_heads'] 
      self.max_actions_per_node = max_actions_per_node 
      self.action_shape = (self.num_nodes, max_actions_per_node)
    else:
      num_attn_heads = 2
      self.max_actions_per_node = max_actions_per_node * self.num_nodes
      self.action_shape =  (max_actions_per_node * self.num_nodes, 1)
      self.action_mask = self.action_mask.flatten()

    concat_obs_width = self.state_obs_width 
    if concat_obs_width % num_attn_heads != 0:
        self.concat_obs_width = ((concat_obs_width // num_attn_heads) + 1) * num_attn_heads
    else:
        self.concat_obs_width = concat_obs_width

    self.action_repeat = action_repeat

    self.importance_scaling = 25

    logging.info('Environment initialised.')


  def take_action(self, cur_pipeline_state: base.State, low_level_goal: jp.array, rng):

    rng, rng1 = jax.random.split(rng)

    low_level_goal = self.get_low_level_goal(low_level_goal, cur_pipeline_state)
    low_level_obs, _ = self.low_level_env.get_obs(cur_pipeline_state, low_level_goal, goal_dist=False) 
    action, _ = self.low_level_policy(low_level_obs, rng1) 
    return self.pipeline_step(cur_pipeline_state, action), rng
  

  def get_low_level_goal(self, g, state):
    
    ldim = (self.position_goals * 12 + self.root_velocity_goals * 6 + self.full_velocity_goals * 12)

    if len(g.shape)==1:
      g = g.reshape((5, ldim + 2*self.goal_importance))
    
    if self.goal_importance:
      importance = g[:, ldim:]
      importance_root = importance[0:1, 1]
      importance_limbs = importance[1:, :].flatten()
      importance = jp.concatenate([importance_root, importance_limbs], axis=0).squeeze().reshape(-1, 1)
      if self.goal_importance_framework == 'continuous':
        importance = (importance + 1)/2
      else:
        importance = jp.where(importance > 0, x=1.0, y=0.0)
        if self.no_root_goals:
          importance = importance.at[0,0].set(0)
    else:
      importance = None

    g = g[:, :ldim]

    if self.position_goals and not self.velocity_goals:
      g = jp.concatenate([g[:, :6], jp.zeros((self.num_nodes, 6)), g[:, 6:], jp.zeros((self.num_nodes, 6))], axis=-1)
    elif self.velocity_goals and not self.position_goals:
      g = jp.concatenate([jp.zeros((self.num_nodes, 6)), g[:, :6], jp.zeros((self.num_nodes, 6)), g[:, 6:]], axis=-1)
    elif self.root_velocity_goals:
      g = jp.concatenate([g, jp.zeros((self.num_nodes, 6))], axis=-1)

    root_goals = g[0:1, :12]
    limb_goals = g[1:, :].reshape(-1, 1, 12).squeeze()

    goals = jp.concatenate([root_goals, limb_goals], axis=-2)
    goals_unnormalised = unnormalize_to_range(
      goals,
      self.low_level_goal_ranges[0],
      self.low_level_goal_ranges[1],
      -1,
      1
    )

    pos_goals = goals_unnormalised[:, 0:3]
    pos_goals = pos_goals.at[0].set(goals_unnormalised[0, 0:3] + state.x.pos[0])
    rot_goals = jax.vmap(spherical_to_quaternion)(goals_unnormalised[:, 3:6])
    vel_goals = goals_unnormalised[:, 6:9]
    ang_goals = goals_unnormalised[:, 9:12]

    x_rel = base.Transform(pos_goals, rot_goals)
    xd_rel = base.Motion(vel_goals, ang_goals)

    return Goal(None, None, None, None, x_rel, xd_rel, importance=importance)


  def get_obs(self, state: base.State) -> jp.ndarray:
    """Return observation input tensor"""
    if hasattr(self, 'state'):
      if self.state == 'below':
        sx, sxd = world_to_relative(state, self.low_level_env)
      elif self.state == 'full':
        sx, sxd = state.x, state.xd
      else:
        state = world_to_egocentric(state)
        state = self.abstract_state(state)
        sx = state.x
        sxd = state.xd
    else:
      if self.state_below:
        sx, sxd = world_to_relative(state, self.low_level_env)
      else:
        state = world_to_egocentric(state)
        state = self.abstract_state(state)
        sx = state.x
        sxd = state.xd

    obs = jp.concatenate([concatenate_attrs(sx), concatenate_attrs(sxd)], axis=-1)

    if self.network_architecture.name == 'MLP':
      obs = obs.reshape(*obs.shape[:-2], -1)

    return obs
  

  def abstract_state(self, state: base.State):

    x, xd = state.x, state.xd
    x_root, xd_root = x.slice(0, 1), xd.slice(0, 1)

    limbs_pos = jp.mean(x.pos[1:].reshape(4, 2, -1), axis=-2)
    limbs_rot = jp.zeros((4, 4)).at[:, 0].set(1)
    limbs_vel = jp.mean(xd.vel[1:].reshape(4, 2, -1), axis=-2)
    limbs_ang = jp.zeros((4, 3))

    x_limbs, xd_limbs = base.Transform(limbs_pos, limbs_rot), base.Motion(limbs_vel, limbs_ang)

    x_rel = x_root.concatenate(x_limbs)
    xd_rel = xd_root.concatenate(xd_limbs)

    return base.State(state.q, state.qd, x_rel, xd_rel, None)
  

 