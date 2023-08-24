import sys
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



class MidLevelEnvV2(PipelineEnv):

  def __init__(
      self,
      low_level_modelpath: str,
      action_repeat = 1,
      distance_reward = 'relative',
      goal_distance_epsilon = 0.001, 
      architecture_configs = DEFAULT_MLP_CONFIGS,
      goal_root_pos_range: jp.ndarray = jp.array([[-10,10], [-10,10], [-0.25, 0.45]]),
      goal_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi], [-jp.pi,jp.pi]]),
      goal_root_vel_range: jp.ndarray = jp.array([[-10,10], [-10,10], [-5, 5]]),
      goal_root_ang_range: jp.ndarray = jp.array([[-5,5], [-5,5], [-10, 10]]),
      goalsampler_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi/12], [-jp.pi,jp.pi]]),
      root_velocity_goals = False,
      obs_mask = None, 
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

    path = epath.resource_path('hct') / f'envs/assets/ant.xml'
    sys = mjcf.load(path)

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
        deterministic=False
    )

    # Agent attributes
    self.dof = low_level_env.dof
    self.num_links = low_level_env.num_links

    # Reward attributes
    self.distance_reward = distance_reward
    
    # Termination attributes
    self._terminate_when_unhealthy = True if distance_reward == 'relative' else False
    self.goal_distance_epsilon = goal_distance_epsilon

    # Reset attributes
    self._reset_noise_scale = 0.1

    # Training attributes
    self.obs_mask = obs_mask
    self.non_actuator_nodes = 0
    self.num_nodes = 5

    # Goal attributes
    self.position_goals = True
    self.velocity_goals = True if root_velocity_goals else False

    self.goal_size = (self.dof*2,)
    self.goal_root_pos_range = goal_root_pos_range
    self.goal_root_rot_range = goal_root_rot_range
    self.goal_root_vel_range = goal_root_vel_range
    self.goal_root_ang_range = goal_root_ang_range

    self.goal_x_mask = 1
    self.goal_xd_mask = jp.zeros((self.num_nodes,3)).at[0].set(1) if self.velocity_goals else False

    # Observation attributes
    self.state_obs_width = 13
    self.goal_obs_width = 7 if not self.velocity_goals else 13

    # Goal sampling attributes
    self.goal_z_cond = jp.array([0.078, 1.6]) if backend == 'generalized' else jp.array([0.08, 1.6])
    self.goal_polar_cond = jp.pi/12
    self.goal_contact_cond = 0.09
    self.air_probability = jp.array([1-0.25, 0.25])
    
    self.limb_ranges = self._get_limb_ranges() 
    self.max_goal_dist = self.limb_ranges['max_dist']
    self.pos_range = self.limb_ranges['pos_range']
    self.rot_range = self.limb_ranges['rot_range']
    self.vel_range_range = self.limb_ranges['vel_range']
    self.ang_range = self.limb_ranges['ang_range']

    goalsampler_q_limit = jp.array(self.sys.dof.limit).T.at[0:3].set(self.goal_root_pos_range).at[3:6].set(goalsampler_root_rot_range)
    goalsampler_qd_limit = self.limb_ranges['goalsampler_qd_limit']
    self.goalsampler_limit = jp.concatenate([goalsampler_q_limit, goalsampler_qd_limit])

    if self.low_level_env.position_goals and self.low_level_env.velocity_goals:
      self.action_mask = jp.ones((self.num_nodes, 4)).at[0].set(0)
      max_actions_per_node = 4
    else:
      self.action_mask = jp.ones((self.num_nodes, 2)).at[0].set(0)
      max_actions_per_node = 2

    self.low_level_goal_ranges = jp.stack(
        [
          jp.stack([self.low_level_env.q_limits[0][6:], self.low_level_env.qd_limits[0][5:]]).T,
          jp.stack([self.low_level_env.q_limits[1][6:], self.low_level_env.qd_limits[1][5:]]).T
        ]
      )
    
    if self.low_level_env.position_goals and not self.low_level_env.velocity_goals:
      self.low_level_goal_ranges = self.low_level_goal_ranges[:, :, 0]

    if self.low_level_env.velocity_goals and not self.low_level_env.position_goals:
      self.low_level_goal_ranges = self.low_level_goal_ranges[:, :, 1]


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

    concat_obs_width = self.state_obs_width + self.goal_obs_width
    if concat_obs_width % num_attn_heads != 0:
        self.concat_obs_width = ((concat_obs_width // num_attn_heads) + 1) * num_attn_heads
    else:
        self.concat_obs_width = concat_obs_width

    self.action_repeat = action_repeat
    self.episode_length = 1000/self.low_level_env.action_repeat

    logging.info('Environment initialised.')


  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale

    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    
    qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(q, qd)

    # Sample and set goal
    goal = self._sample_goal(rng3, pipeline_state) 

    # Get observation
    obs, (goal_dist_egocentric_frame, goal_dist_world_frame) = self.get_obs(pipeline_state, goal)
    
    # Set metrics
    reward, done = jp.zeros(2)

    if self.distance_reward == 'absolute':
      reward= -goal_dist_world_frame

    metrics = {
      'reward': reward,
      'goal_dist_egocentric_frame': goal_dist_egocentric_frame,
      'goal_dist_egocentric_frame_normalised': goal_dist_egocentric_frame/self.max_goal_dist,
      'goal_dist_world_frame': goal_dist_world_frame,
      'is_unhealthy': 0.0,
      'goal_reached': 0.0
    }

    info = {
      'goal': goal,
      'rng': rng
      }

    return State(pipeline_state, obs, reward, done, metrics, info)


  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    if action.shape[-1] == 1:
        action = jp.squeeze(action, axis=-1) 

    rng = state.info['rng']
    goal = state.info['goal']
    prev_goal_dist_world_frame = state.metrics['goal_dist_world_frame']

    rng, rng1 = jax.random.split(rng)

    # Take action
    prev_pipeline_state = state.pipeline_state
    low_level_goal = self._get_low_level_goal(action) 
    low_level_obs, _ = self.low_level_env.get_obs(prev_pipeline_state, low_level_goal)
    action, _ = self.low_level_policy(low_level_obs, rng1) 
    pipeline_state = self.pipeline_step(prev_pipeline_state, action) 

    # Compute state observation
    obs, (goal_dist_egocentric_frame, goal_dist_world_frame) = self.get_obs(pipeline_state, goal)
    goal_dist_egocentric_frame_normalised = goal_dist_egocentric_frame/self.max_goal_dist

    # Check if unhealthy
    is_unhealthy = jp.where(rotate(jp.array([0, 0, 1]), pipeline_state.x.rot[0])[-1] < 0, x=1.0, y=0.0)

    # Check if goal reached
    goal_reached = jp.where(
      goal_dist_egocentric_frame_normalised < self.goal_distance_epsilon, x=1.0, y=0.0
    )

    # Compute rewards: 
    if self.distance_reward == 'absolute':
      reward = -goal_dist_world_frame
    else:
      reward = (prev_goal_dist_world_frame - goal_dist_world_frame)/self.dt

    if self._terminate_when_unhealthy:
      done = 0.0 + jp.logical_or(is_unhealthy, goal_reached)
    else:
      done = 0.0 + goal_reached

    state.metrics.update(
      reward=reward,
      goal_dist_egocentric_frame=goal_dist_egocentric_frame,
      goal_dist_egocentric_frame_normalised=goal_dist_egocentric_frame_normalised,
      goal_dist_world_frame=goal_dist_world_frame,
      is_unhealthy=1.0*is_unhealthy,
      goal_reached=1.0*goal_reached
    )

    state.info.update(
      rng=rng
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
  
  def _get_low_level_goal(self, g):

    if len(g.shape)==1:
      g = g.reshape((self.num_nodes, 2 * (self.low_level_env.position_goals + self.low_level_env.velocity_goals)))
    
    root_goals = jp.zeros((1, self.low_level_env.position_goals + self.low_level_env.velocity_goals))
    limb_goals = g[1:, :].reshape(2*(self.num_nodes-1), self.low_level_env.position_goals + self.low_level_env.velocity_goals)

    goals = jp.concatenate([root_goals, limb_goals], axis=0).squeeze()

    if self.low_level_env.position_goals and not self.low_level_env.velocity_goals:
      ja = goals[0] * self.low_level_env.position_goals
      jv = goals[-1] * self.low_level_env.velocity_goals
    else:
      ja = goals[..., 0] * self.low_level_env.position_goals
      jv = goals[..., -1] * self.low_level_env.velocity_goals

    return Goal(None, None, None, None, None, None, ja, jv, None)


  def get_obs(self, state: base.State, goal: Goal) -> jp.ndarray:
    """Return observation input tensor"""
    
    def _mask_root(x):
      root_x = x.take(0)
      mask = base.Transform(jp.array([0.0, 0.0, 1.0]), jp.array([1.0, 1.0, 1.0, 1.0]))
      root_x = mul(root_x, mask)
      return x.index_set(0, root_x)
    
    abstract_world_state = self.abstract_state(state)

    state = world_to_egocentric(state)
    state = self.abstract_state(state)

    sx = state.x
    sxd = state.xd

    if self.position_goals:
      gx = goal.x_rel.vmap(in_axes = (0, 0)).to_local(sx)
    else:
      gx = base.Transform(jp.empty((self.num_nodes, 0)), jp.empty((self.num_nodes, 0)))

    if self.velocity_goals:
      gxd = jax.tree_map(lambda x: jax.vmap(inv_rotate)(x, sx.rot) * self.goal_xd_mask , goal.xd_rel.__sub__(sxd)) 
    else:
      gxd = base.Motion(jp.empty((self.num_nodes, 0)), jp.empty((self.num_nodes, 0)))

    s_obs = jp.concatenate([concatenate_attrs(_mask_root(sx)), concatenate_attrs(sxd)], axis=-1)
    g_obs = jp.concatenate([concatenate_attrs(gx), concatenate_attrs(gxd)], axis=-1)

    obs = jp.concatenate([s_obs, g_obs], axis=-1)
    obs = pad(obs, self.concat_obs_width)

    if self.network_architecture.name == 'MLP':
      obs = obs.reshape(*obs.shape[:-2], -1)

    goal_dist_world_frame = self._dist(abstract_world_state.x, abstract_world_state.xd, goal.x, goal.xd)
    goal_dist_egocentric_frame = self._dist(state.x, state.xd, goal.x_rel, goal.xd_rel)

    return obs, (goal_dist_egocentric_frame, goal_dist_world_frame)
  

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
    g = jax.random.uniform(
      rng, 
      shape=self.goal_size, 
      minval=self.goalsampler_limit[:, 0], 
      maxval=self.goalsampler_limit[:, 1]
    )
    
    q = q_spherical_to_quaternion(g[:self.dof], state, self.sys)
    qd = g[-self.dof:]
    x, xd = forward(self.sys, q, qd)

    z = x.pos[:,2]
    z = z - jp.min(z) + self.goal_z_cond[0] 
    z = z + jax.random.choice(rng1, jp.array([0, 1]), p=self.air_probability) * jax.random.uniform(rng2, minval=0, maxval=self.goal_z_cond[1] - z[0])

    q = q.at[2].set(z[0])
    x, xd = forward(self.sys, q, qd)

    goal_state = base.State(q, qd, x, xd, None)
    abstract_world_state = self.abstract_state(goal_state)
    goal_state = world_to_egocentric(goal_state)
    abstract_state = self.abstract_state(goal_state)

    return Goal(q, qd, abstract_world_state.x, abstract_world_state.xd, abstract_state.x, abstract_state.xd)
  

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
  

  def _dist(self, state_x, state_xd, goal_x, goal_xd):
    return dist(self, state_x, state_xd, goal_x, goal_xd)[0]


  def _get_limb_ranges(self):

    filepath = '/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/envs/ranges/'

    env = 'Mid level v2'

    if self.velocity_goals:
      velocity_goals = 'root'
    else:
      velocity_goals = False
    
    position_goals = str(self.position_goals)

    variant = f'{env}, {position_goals}, {velocity_goals}'
    
    filename = f'{filepath}/{variant}'
    
    if os.path.isfile(filename):
      return load(filename)

    quaternion_to_spherical_vmap = jax.vmap(quaternion_to_spherical, in_axes=0)

    test_rollout, _ = AntTest().test_rollout()

    rollout_rel = [self.abstract_state(state) for state in test_rollout]
    rollout_rel_x = [t.x for t in rollout_rel]
    rollout_rel_xd = [t.xd for t in rollout_rel]

    rollout_rel_pos = jp.stack([x.pos for x in rollout_rel_x])
    rollout_rel_rot = {
      'quaternion': jp.stack([x.rot for x in rollout_rel_x]),
      'spherical': jp.stack([quaternion_to_spherical_vmap(x.rot) for x in rollout_rel_x])
    }
    rollout_rel_vel = jp.stack([xd.vel for xd in rollout_rel_xd])
    rollout_rel_ang = jp.stack([xd.ang for xd in rollout_rel_xd])
    rollout_qd = jp.stack([state.qd for state in test_rollout])

    # goal ranges
    pos_ranges = (
      jp.min(rollout_rel_pos, axis=0).at[0].set(self.goal_root_pos_range[:,0]), 
      jp.max(rollout_rel_pos, axis=0).at[0].set(self.goal_root_pos_range[:,1])
    )
    rot_ranges = minmax_angles(rollout_rel_rot['spherical'], self.goal_root_rot_range)
    vel_ranges = (
      jp.min(rollout_rel_vel, axis=0).at[0].set(self.goal_root_vel_range[:,0]), 
      jp.max(rollout_rel_vel, axis=0).at[0].set(self.goal_root_vel_range[:,1])
    )
    ang_ranges = (
      jp.min(rollout_rel_ang, axis=0).at[0].set(self.goal_root_ang_range[:,0]), 
      jp.max(rollout_rel_ang, axis=0).at[0].set(self.goal_root_ang_range[:,1])
    )
    goalsampler_qd_limit = jp.array(
      [jp.min(rollout_qd, axis=0).at[0:6].set(jp.concatenate([self.goal_root_vel_range[:,0], self.goal_root_ang_range[:,0]])), 
      jp.max(rollout_qd, axis=0).at[0:6].set(jp.concatenate([self.goal_root_vel_range[:,1], self.goal_root_ang_range[:,1]]))]
    ).T

    # max distances
    x_min = rollout_rel_x[7*30-1].index_set(0, base.Transform(self.goal_root_pos_range[:,0], jp.array([1,0,0,0])))
    x_max = rollout_rel_x[15*30-1].index_set(0, base.Transform(self.goal_root_pos_range[:,1], jp.array([0,1,0,0])))
    xd_min = base.Motion(vel_ranges[0], ang_ranges[0])
    xd_max = base.Motion(vel_ranges[1], ang_ranges[1])


    max_dist = self._dist(x_min, xd_min, x_max, xd_max)

    return_dict = {
      'max_dist': max_dist,
      'pos_range': pos_ranges, 
      'rot_range': rot_ranges, 
      'vel_range': vel_ranges, 
      'ang_range': ang_ranges, 
      'goalsampler_qd_limit': goalsampler_qd_limit,
    }

    save(filename, return_dict)
    return return_dict
  
"""
if self.position_goals:
  self.goal_x_mask = 1
  goal_x_obs_width = 7
  x_action_mask = jp.ones((5,12)).at[0,6:].set(jp.zeros(6))
else:
  self.goal_x_mask = 0
  self.goal_x_obs_width = 0
  x_action_mask = jp.empty((5,0))

if self.velocity_goals:
  goal_xd_obs_width = 6
  if self.full_velocity_goals:
    self.goal_xd_mask = 1
    xd_action_mask = jp.ones((5,12)).at[0,6:].set(jp.zeros(6))
  else:
    self.goal_xd_mask = jp.zeros((self.num_links,3)).at[0].set(1.0)
    xd_action_mask = jp.zeros((5,6)).at[0].set(jp.ones(6))
else:
  goal_xd_obs_width = 0
  self.xd_mask = 0
"""
