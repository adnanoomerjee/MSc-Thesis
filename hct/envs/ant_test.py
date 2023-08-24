## pylint:disable=g-multiple-import
"""Creates an environment for the lowest level of a hierarchical framework"""
import sys
import inspect
import os

from hct.envs.goal import Goal
from hct.envs.tools import *
from hct.training.configs import NetworkArchitecture, SMALL_TRANSFORMER_CONFIGS, DEFAULT_MLP_CONFIGS
from hct.io import model, html

from brax import base, generalized, math
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward

from etils import epath

import jax
from jax import numpy as jp

from typing import Optional, Literal, Tuple

from absl import logging


class AntTest(PipelineEnv):


  def __init__(
      self,
      ctrl_cost_weight=0.5,
      use_contact_forces=False,
      contact_cost_weight=5e-4,
      healthy_reward=1.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.2, 1.0),
      contact_force_range=(-1.0, 1.0),
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      backend='positional',
      architecture_configs=None,
      **kwargs,
  ):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')
  
    path = epath.resource_path('hct') / f'envs/assets/ant.xml'
    sys = mjcf.load(path)

    n_frames = 5

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

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._contact_force_range = contact_force_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

    # Training attributes
    self.obs_mask = None
    self.non_actuator_nodes = None
    self.action_mask = None
    self.num_nodes = 9

    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name='MLP', **DEFAULT_MLP_CONFIGS)
    self.max_actions_per_node = 1 if self.network_architecture.name=='Transformer' else 8

    self.action_repeat = 1
    self.horizon = 72
    self.episode_length = 1000
    self.action_shape = (8, 1)


    self.episode_length = 1000
    if self._use_contact_forces:
      raise NotImplementedError('use_contact_forces not implemented.')

  def reset(self, rng) -> State:
    """Resets the environment to an initial state."""


    pipeline_state = self._sample_state(rng)
    obs = self._get_obs(pipeline_state)

    reward, done, zero = jp.zeros(3)

    zeros = jp.zeros((self.action_size,))
    metrics = {
      'reward': zero
    }

    info = {
      'rng': jax.random.split(rng)[0],
      'min_ja': zeros,
      'max_ja': zeros,
      'min_jv': zeros,
      'max_jv': zeros
    }

    return State(pipeline_state, obs, reward, done, metrics, info)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    rng, rng1 = jax.random.split(state.info['rng'])
    prev_min_ja, prev_max_ja = state.info['min_ja'], state.info['max_ja']
    prev_min_jv, prev_max_jv = state.info['min_jv'], state.info['max_jv']

    is_healthy = jp.where(rotate(jp.array([0, 0, 1]), pipeline_state.x.rot[0])[-1] > 0, x=1.0, y=0.0)

    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ja = state.pipeline_state.q[7:]
    jv = state.pipeline_state.qd[6:]

    min_ja = jp.where(ja<prev_min_ja, x=ja, y=prev_min_ja)
    max_ja = jp.where(ja>prev_max_ja, x=ja, y=prev_max_ja)
    min_jv = jp.where(jv<prev_min_jv, x=jv, y=prev_min_jv)
    max_jv = jp.where(jv>prev_max_jv, x=jv, y=prev_max_jv)

    abs_cur = jp.abs(min_ja) + jp.abs(max_ja) + jp.abs(min_jv) + jp.abs(max_jv)
    abs_prev = jp.abs(prev_min_ja) + jp.abs(prev_max_ja) + jp.abs(prev_min_jv) + jp.abs(prev_max_jv)

    diff = abs_cur - abs_prev
    reward = jp.sum(jp.square(jp.where(diff > 0, abs_cur, 0)))

    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

    pipeline_state  = jax.lax.cond(done, self._sample_state, lambda x: pipeline_state, rng1)
    obs = self._get_obs(pipeline_state)

    state.metrics.update(
      reward=reward
    )
    state.info.update(
      rng=rng,
      min_ja=min_ja,
      max_ja=max_ja,
      min_jv=min_jv,
      max_jv=max_jv)
    
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
    """Observe ant body position and velocities."""
    qpos = pipeline_state.q
    qvel = pipeline_state.qd

    if self._exclude_current_positions_from_observation:
      qpos = pipeline_state.q[2:]

    return jp.concatenate([qpos] + [qvel])
  
  def _sample_state(self, rng: jp.ndarray):
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

    rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale

    q = self.sys.init_q + jax.random.uniform(
      rng3, (self.sys.q_size(),), minval=low, maxval=hi
    )

    qd = hi * jax.random.normal(rng4, (self.sys.qd_size(),))

    return self.pipeline_init(q, qd)

  def move_limbs(self, limb_ids, actuator_force):
    return jp.zeros((self.action_size,)).at[jp.array(limb_ids)].set(actuator_force)
  
  
  def test_rollout(self):
        
    path = epath.resource_path('hct') / f'envs/assets/ant_test.xml'
    self.sys = mjcf.load(path)

    def reset(limit_id: int) -> State:
      """Resets the environment to an initial state."""

      joint_angles = jax.lax.select(limit_id==0, jp.array(self.sys.dof.limit[0]), jp.array(self.sys.dof.limit[1]))
      q = jp.concatenate([jp.array([0,0,0,1,0,0,0]), joint_angles[6:]])
      qd = jp.zeros((self.sys.qd_size(),))
      
      if limit_id is None:
        q = self.sys.init_q

      pipeline_state = self.pipeline_init(q, qd)
      obs = self._get_obs(pipeline_state)

      reward, done, zero = jp.zeros(3)
      metrics = {
          'reward_forward': zero,
          'reward_survive': zero,
          'reward_ctrl': zero,
          'reward_contact': zero,
          'x_position': zero,
          'y_position': zero,
          'distance_from_origin': zero,
          'x_velocity': zero,
          'y_velocity': zero,
          'forward_reward': zero,
      }
      return State(pipeline_state, obs, reward, done, metrics)

    def step(state: State, action: jp.ndarray) -> State:
      """Run one timestep of the environment's dynamics."""
      pipeline_state0 = state.pipeline_state
      pipeline_state = self.pipeline_step(pipeline_state0, action)

      velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
      forward_reward = velocity[0]

      min_z, max_z = self._healthy_z_range
      is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
      is_healthy = jp.where(
          pipeline_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy
      )
      if self._terminate_when_unhealthy:
        healthy_reward = self._healthy_reward
      else:
        healthy_reward = self._healthy_reward * is_healthy
      ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
      contact_cost = 0.0

      obs = self._get_obs(pipeline_state)
      reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
      done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
      state.metrics.update(
          reward_forward=forward_reward,
          reward_survive=healthy_reward,
          reward_ctrl=-ctrl_cost,
          reward_contact=-contact_cost,
          x_position=pipeline_state.x.pos[0, 0],
          y_position=pipeline_state.x.pos[0, 1],
          distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
          x_velocity=velocity[0],
          y_velocity=velocity[1],
          forward_reward=forward_reward,
      )
      return state.replace(
          pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
      )


    filename = '/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/envs/ranges/test_rollout'

    if os.path.isfile(filename):
      return model.load(filename)

    jit_env_reset = jax.jit(reset)
    jit_env_step = jax.jit(step)
    jit_move_limbs = jax.jit(self.move_limbs)

    state = jit_env_reset(limit_id=None)
    rollout = [state.pipeline_state]

    limb_movements = (
      (list(range(1,8,2)), -1),
      (list(range(1,8,2)), 1),
      (list(range(1,8,2)), -1),
      (list(range(0,8,2)), -1),
      (list(range(0,8,2)), 1),
      (list(range(0,8,2)), -1),
      (list(range(1,8,2)), 1),
      (list(range(1,8,2)), -1),
      (list(range(1,8,2)), 1),
      (list(range(0,8,2)), 1),
      (list(range(0,8,2)), -1),
      (list(range(0,8,2)), 1),
      (list(range(1,8,2)), -1),
      (list(range(1,8,2)), 1),
      (list(range(1,8,2)), -1),
    )

    for args in limb_movements:
      for _ in range(30):
          act = jit_move_limbs(*args)
          state = jit_env_step(state, act)
          rollout.append(state.pipeline_state)

    output = (rollout, html.render(self.sys.replace(dt=self.dt), rollout))

    model.save(filename, output)
    path = epath.resource_path('hct') / f'envs/assets/ant.xml'
    self.sys = mjcf.load(path)

    return output





'''  
  def _limb_dist(
      self, 
      state1: base.State, 
      state2: base.State, 
      limb_id: int,  
      frame: Literal['world', 'relative']):
    """
    Computes distance d(s,g) between state and goal in world frame, 
    accounting for quaternion double cover.

    dist(s,g) = ||s-g||
    """
    if frame == 'world':
        state1_x = state1.x.take(limb_id)
        state1_xd = state1.xd.take(limb_id) 
        state2_x = state2.x.take(limb_id)
        state2_xd = state2.xd.take(limb_id)
    else:
        state1_x = world_to_relative(state1.x.take(limb_id), self.sys)
        state1_xd = world_to_relative(state1.xd.take(limb_id), self.sys)
        state2_x = world_to_relative(state2.x.take(limb_id), self.sys)
        state2_xd = world_to_relative(state2.xd.take(limb_id), self.sys)

    rpos = state1_x.pos - state2_x.pos
    rrot = dist_quat(state1_x.rot, state2_x.rot)
    rx = concatenate_attrs(base.Transform(rpos, rrot))
    rxd = concatenate_attrs(state1_xd.__sub__(state2_xd))
    x_dist = safe_norm(rx) 
    xd_dist = safe_norm(rxd)
    return x_dist, xd_dist
'''


'''
  def get_limb_x_dist(self, x0: base.Transform, x1: base.Transform, limb_id):
    x0 = x0.take(limb_id)
    x1 = x1.take(limb_id)
    rpos = x0.pos - x1.pos
    rrot = dist_quat(x0.rot, x1.rot)
    rx = concatenate_attrs(base.Transform(rpos, rrot))
    x_dist = safe_norm(rx) 
    return x_dist

  def get_limb_xd_dist(self, xd0: base.Transform, xd1: base.Transform, limb_id):
    xd0 = xd0.take(limb_id)
    xd1 = xd1.take(limb_id)
    rxd = concatenate_attrs(xd0.__sub__(xd1))   
    return safe_norm(rxd)
    

  def get_limb_ranges(self):

    test_rollout, html = self.test_rollout()
    
    quaternion_to_spherical_vmap = jax.vmap(quaternion_to_spherical, in_axes=0)
    jit_env_reset = jax.jit(self.reset)
    jit_env_step = jax.jit(self.step)
    jit_move_limbs = jax.jit(self.move_limbs)

    q0 = jp.concatenate([jp.array([0,0,0,1,0,0,0]), jp.array(self.sys.dof.limit[0])[6:]])
    q1 = jp.concatenate([jp.array([0,0,0,1,0,0,0]), jp.array(self.sys.dof.limit[1])[6:]])

    qd0 = jp.zeros((self.sys.qd_size(),))   

    x0 = world_to_relative(forward(self.sys, q0, qd0)[0], self.sys)
    x1= world_to_relative(forward(self.sys, q1, qd0)[0], self.sys)

    xd0 = base.Motion.zero(shape = (self.sys.num_links(),))
    xd0 = world_to_relative(xd0, self.sys)

    upper_leg_dists = {}
    lower_leg_dists = {}

    upper_leg_dists['max_x_dist'] = self.get_limb_x_dist(x0, x1, 1)
    lower_leg_dists['max_x_dist'] = self.get_limb_x_dist(x0, x1, 2)

    rollout = []
    rollout_rel_x = []
    rollout_rel_xd = []

    '''    
'''
for ranges in (upper_leg_dists, lower_leg_dists):

  if ranges == upper_leg_dists:
    limb_id = 3
  else:
    limb_id = 4

  ranges['max_xd_dist'] = 0'''
'''
    for limit_id in (0,1):

      for actuator_force in (-1,1,-1,1):

        state = jit_env_reset(limit_id=limit_id)

        for _ in range(30):

          rollout.append(state.pipeline_state)

          pipeline_state_rel_x = world_to_relative(state.pipeline_state.x, self.sys)
          pipeline_state_rel_xd = world_to_relative(state.pipeline_state.xd, self.sys)
          rollout_rel_x.append(pipeline_state_rel_x)
          rollout_rel_xd.append(pipeline_state_rel_xd)

          xd_dist = self.get_limb_xd_dist(xd0, pipeline_state_rel_xd , limb_id)
          if xd_dist > ranges['max_xd_dist']:
            ranges['max_xd_dist'] = xd_dist

          act = jit_move_limbs(limb_id=limb_id, actuator_force=actuator_force)
          state = jit_env_step(state, act)

    rollout_rel_pos = jp.stack([x.pos for x in rollout_rel_x])
    rollout_rel_rot = jp.stack([quaternion_to_spherical_vmap(x.rot) for x in rollout_rel_x])
    rollout_rel_vel = jp.stack([xd.vel for xd in rollout_rel_xd])
    rollout_rel_ang = jp.stack([xd.ang for xd in rollout_rel_xd])

    pos_ranges = jp.min(rollout_rel_pos, axis=0), jp.max(rollout_rel_pos, axis=0)
    rot_ranges = jp.min(rollout_rel_rot, axis=0), jp.max(rollout_rel_rot, axis=0)
    vel_ranges = jp.min(rollout_rel_vel, axis=0), jp.max(rollout_rel_vel, axis=0)
    ang_ranges = jp.min(rollout_rel_ang, axis=0), jp.max(rollout_rel_ang, axis=0)

    return_dict = {
      'upper_leg_dists': upper_leg_dists, 
      'lower_leg_dists': lower_leg_dists,
      'pos_ranges': pos_ranges, 
      'rot_ranges': rot_ranges, 
      'vel_ranges': vel_ranges, 
      'ang_ranges': ang_ranges, 
      'rollout': rollout,
      'rollout_rel_x': rollout_rel_x,
      'rollout_rel_xd': rollout_rel_xd,
      'html': html.render(self.sys.replace(dt=self.dt), rollout)
    }
    return return_dict'''




