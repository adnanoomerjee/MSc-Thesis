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



class MidLevelEnv(PipelineEnv):

  def __init__(
      self,
      low_level_modelpath: str,
      action_repeat=1,
      goal_root_pos_range: jp.ndarray = jp.array([[-6, 6], [-6, 6], [-0.25, 0.45]]),
      goal_distance_epsilon = 0.1, # current best 0.01
      architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
      goal_dist = 'absolute',
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

    sys = low_level_env.sys

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
    self.link_parents = (-1, 0, 0, 0, 0)

    # Reward attributes
    self.distance_reward = low_level_env.distance_reward
    self.unhealthy_cost = low_level_env.unhealthy_cost
    self.ctrl_cost = low_level_env.ctrl_cost
    self.reward_goal_reached = low_level_env.reward_goal_reached

    # Termination attributes
    self._terminate_when_unhealthy = low_level_env._terminate_when_unhealthy
    self._terminate_when_goal_reached = low_level_env._terminate_when_goal_reached
    self.goal_distance_epsilon = goal_distance_epsilon * 0.1 if goal_dist == 'relative' else goal_distance_epsilon

    self.rot_dist = True

    # Reset attributes
    self._reset_noise_scale = low_level_env._reset_noise_scale

    # Training attributes
    self.obs_mask = low_level_env.obs_mask
    self.non_actuator_nodes = None
    self.num_nodes = 5

    # Goal attributes
    self.goal_nodes = low_level_env.goal_nodes
    self.position_goals = low_level_env.position_goals
    self.velocity_goals = low_level_env.velocity_goals
    self.root_velocity_goals = low_level_env.root_velocity_goals
    self.full_velocity_goals = low_level_env.full_velocity_goals
    self.goal_size = low_level_env.goal_size
    self.goal_root_pos_range = goal_root_pos_range
    self.goal_root_rot_range = low_level_env.goal_root_rot_range
    self.goal_root_vel_range = low_level_env.goal_root_vel_range
    self.goal_root_ang_range = low_level_env.goal_root_ang_range
    self.goal_x_mask = low_level_env.goal_x_mask[:5] if hasattr(low_level_env.goal_x_mask, 'shape') else low_level_env.goal_x_mask
    self.goal_xd_mask = low_level_env.goal_xd_mask[:5] if hasattr(low_level_env.goal_xd_mask, 'shape') else low_level_env.goal_xd_mask
    self.root_mask = self.low_level_env.root_mask
    self.goal_importance = self.low_level_env.goal_importance
    self.goal_importance_framework = self.low_level_env.goal_importance_framework
    self.importance_scaling = 40
    self.goal_dist = goal_dist

    # Observation attributes
    self.state_obs_width = low_level_env.state_obs_width
    self.goal_obs_width = low_level_env.goal_obs_width

    # Goal sampling attributes
    self.goal_z_cond = low_level_env.goal_z_cond
    self.goal_polar_cond = low_level_env.goal_polar_cond
    self.goal_contact_cond = low_level_env.goal_contact_cond
    self.air_probability = low_level_env.air_probability

    self.goalsampler_limit = low_level_env.goalsampler_limit 

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

    self.limb_ranges = self._get_limb_ranges() 
    self.max_goal_dist = self.limb_ranges['max_dist']
    self.max_root_goal_dist = self.limb_ranges['max_root_dist']
    self.pos_range = self.limb_ranges['pos_range']
    self.rot_range = self.limb_ranges['rot_range']
    self.vel_range_range = self.limb_ranges['vel_range']
    self.ang_range = self.limb_ranges['ang_range']
    self.max_sq_dist_nodes = self.limb_ranges['max_sq_dist_nodes']
    self.minmax = self.limb_ranges['minmax']
    
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
    self.horizon = 100
    self.reset_interval = self.low_level_env.reset_interval
    self.episode_length = self.horizon * self.reset_interval
    self.resample_probability=self.low_level_env.resample_probability

    logging.info('Environment initialised.')


  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    rng, rng1, rng2 = jax.random.split(rng, 3)

    pipeline_state = self._sample_state(rng1)

    # Sample and set goal
    goal = self._sample_goal(rng2, pipeline_state) 

    # Get observation
    obs, goal_dist = self.get_obs(pipeline_state, goal, goal_dist=True)
    
    # Set metrics
    reward, done = jp.zeros(2)

    if self.distance_reward == 'absolute':
      reward= -goal_dist

    metrics = {
      'reward': reward,
      'goal_dist':goal_dist,
      'is_unhealthy': 0.0,
      'goals_reached': 0.0,
      'weight': 1.0,
      'cumulative_final_goal_dist': 0.0,
      'cumulative_return': 0.0
    }

    info = {
      'goal': goal,
      'goal_count': 0.0,
      'rng': rng,
      'sub_episode_step': 0.0,
      'running_return': 0.0,
      'running_final_goal_dist': 0.0
    }

    return State(pipeline_state, obs, reward, done, metrics, info)


  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    if action.shape[-1] == 1:
      action = jp.squeeze(action, axis=-1) 

    goal = state.info['goal']
    goal_count = state.info['goal_count']
    rng = state.info['rng']
    prev_goal_dist = state.metrics['goal_dist']
    sub_episode_step = state.info['sub_episode_step']

    weight = state.metrics['weight']
    running_return = state.info['running_return']
    running_final_goal_dist = state.info['running_final_goal_dist']

    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    pipeline_state = state.pipeline_state

    # Check if goal reached
    goal_reached = jp.where(
      prev_goal_dist <= self.goal_distance_epsilon * (jp.sqrt(jp.sum(4 * self.num_nodes * goal.importance))), x=1.0, y=0.0
    )

    # Check if unhealthy
    is_unhealthy = jp.where(rotate(jp.array([0, 0, 1]), pipeline_state.x.rot[0])[-1] < 0, x=1.0, y=0.0)

    random_resample_goal = jax.random.choice(rng1, jp.array([0, 1]), p=jp.array([1-self.resample_probability, self.resample_probability]))
    
    done = jp.where(
      is_unhealthy, x=1.0, y=jp.where(goal_count >= self.reset_interval, x=1.0, y=0.0))
    
    resample_goal = jp.where(
      random_resample_goal > 0.5, x=1.0, y=jp.where(jp.logical_or(done, goal_reached), 
                                                    x=1.0, y=jp.where(sub_episode_step >= self.horizon, x=1.0, y=0.0)))
    
    weight = weight * (1 - resample_goal) + resample_goal * (goal_count + 1)

    running_return = running_return + state.reward
    running_final_goal_dist = running_final_goal_dist + prev_goal_dist * (resample_goal)

    cumulative_return = state.metrics['cumulative_return'] * (1 - done) + done * running_return
    cumulative_final_goal_dist = state.metrics['cumulative_final_goal_dist'] * (1 - done) + done * running_final_goal_dist

    running_final_goal_dist = running_final_goal_dist * (1 - done)
    running_return = running_return * (1 - done)

    sub_episode_step = jp.where(resample_goal, x=0.0, y=sub_episode_step + 1)
    goal_count = jp.where(done, x=0.0, y=goal_count + resample_goal)
    goals_reached = (state.metrics['goals_reached'] + goal_reached)/(goal_count+1)

     # Take action
    pipeline_state = jax.lax.cond(done, rng2, lambda x: self._sample_state(x), (pipeline_state, action, rng2), lambda x: self.take_action(x[0], x[1], x[2]))
    goal = jax.lax.cond(resample_goal, self._sample_goal, lambda x, y: goal, rng3, pipeline_state)

    # Compute state observation
    obs, goal_dist = self.get_obs(pipeline_state, goal, goal_dist=True)

    # Compute rewards: 
    if self.distance_reward == 'absolute':
      reward = -goal_dist
    else:
      reward = (prev_goal_dist - goal_dist)/self.dt

    reward = reward * (1 - done) * (1 - resample_goal) + is_unhealthy * self.unhealthy_cost 

    state.info.update(
      goal=goal,
      goal_count=goal_count,
      rng=rng,
      sub_episode_step=sub_episode_step,
      running_final_goal_dist=running_final_goal_dist,
      running_return=running_return
    )

    state.metrics.update(
      reward=reward,
      goal_dist=goal_dist,
      cumulative_final_goal_dist=cumulative_final_goal_dist,
      cumulative_return=cumulative_return,
      weight=weight,
      is_unhealthy=1.0*is_unhealthy,
      goals_reached=goals_reached
    )

    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
  
  def take_action(self, pipeline_state, action, rng):
    # Take action
    low_level_goal = self.get_low_level_goal(action)
    low_level_obs, _ = self.low_level_env.get_obs(pipeline_state, low_level_goal, goal_dist=False) 
    action, _ = self.low_level_policy(low_level_obs, rng) 
    return self.pipeline_step(pipeline_state, action)
  
  
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


  def get_obs(self, state: base.State, goal: Goal, goal_dist = True) -> jp.ndarray:
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
      rot = gx.rot.at[1:].set(jp.array([1, 0, 0, 0]))
      gx = base.Transform(gx.pos, rot)
    else:
      gx = base.Transform(jp.empty((self.num_nodes, 0)), jp.empty((self.num_nodes, 0)))

    if self.velocity_goals:
      gxd = jax.tree_map(lambda x: jax.vmap(inv_rotate)(x, sx.rot) * self.goal_xd_mask , goal.xd_rel.__sub__(sxd)) 
    else:
      gxd = base.Motion(jp.empty((self.num_nodes, 0)), jp.empty((self.num_nodes, 0)))

    importance = goal.importance if self.goal_importance else jp.empty((self.num_nodes, 0))

    s_obs = jp.concatenate([concatenate_attrs(_mask_root(sx)), concatenate_attrs(sxd)], axis=-1)
    g_obs = jp.concatenate([concatenate_attrs(gx), concatenate_attrs(gxd), importance], axis=-1)

    obs = jp.concatenate([s_obs, g_obs], axis=-1)
    obs = pad(obs, self.concat_obs_width)

    if self.network_architecture.name == 'MLP':
      obs = obs.reshape(*obs.shape[:-2], -1)

    if goal_dist:
      if self.goal_dist == 'relative':
        goal_dist = self._dist(state.x, state.xd, goal.x_rel, goal.xd_rel, importance=goal.importance)
      else:
        goal_dist = self._dist(abstract_world_state.x, abstract_world_state.xd, goal.x, goal.xd, importance=goal.importance)
    else:
      goal_dist = None

    return obs, goal_dist
  

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

    rng, rng1, rng2 ,rng4 = jax.random.split(rng, 4)
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

    if self.goal_importance:
      importance = jax.nn.sigmoid(jax.random.uniform(rng4, shape=(self.num_nodes, 1), minval = -40, maxval = 10))
    else:
      importance = 1

    return Goal(q, qd, abstract_world_state.x, abstract_world_state.xd, abstract_state.x, abstract_state.xd, importance=importance)


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

    return state.replace(x=x_rel, xd=xd_rel)
  

  def _dist(self, state_x, state_xd, goal_x, goal_xd, importance):
    return dist(self, state_x, state_xd, goal_x, goal_xd, importance=importance)[0]


  def _get_limb_ranges(self):

    filepath = '/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/envs/ranges/'

    env = 'Mid level'

    if self.root_velocity_goals:
      velocity_goals = 'root'
    elif self.full_velocity_goals:
      velocity_goals = 'full'
    else:
      velocity_goals = 'False'
    
    position_goals = str(self.position_goals)
    rot_dist = str(self.rot_dist)

    variant = f'{env}, {position_goals}, {velocity_goals}, {rot_dist}, {self.goal_dist}'
    
    filename = f'{filepath}/{variant}'
    
    if os.path.isfile(filename):
      return load(filename)

    quaternion_to_spherical_vmap = jax.vmap(quaternion_to_spherical, in_axes=0)

    test_rollout, _ = AntTest().test_rollout()

    rollout_rel = [self.abstract_state(state) for state in test_rollout]
    rollout_rel_x = [state.x for state in rollout_rel]
    rollout_rel_xd = [state.xd for state in rollout_rel]

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
    x_min = base.Transform(pos_ranges[0], pos_ranges[0])
    x_max = base.Transform(pos_ranges[1], pos_ranges[1])
    xd_min = base.Motion(vel_ranges[0], ang_ranges[0])
    xd_max = base.Motion(vel_ranges[1], ang_ranges[1])

    minmax = {}

    minmax['pos'] = jp.abs(x_min.pos - x_max.pos)# if self.goal_dist == 'relative' else 12
    minmax['rot'] = jax.vmap(dist_quat)(x_min.rot, x_max.rot)
    minmax['vel'] = jp.abs(safe_norm(xd_min.vel) - safe_norm(xd_max.vel))# if self.goal_dist == 'relative' else 10
    minmax['ang'] = jp.abs(xd_min.ang - xd_max.ang)

    sq_dist_node = max_sq_dist_nodes(self, x_min, xd_min, x_max, xd_max, root_dist=True)
    max_root_dist = jp.sqrt(sq_dist_node[0])
    max_dist = jp.sqrt(self.num_nodes)
    
    return_dict = {
      'minmax': minmax,
      'max_dist': max_dist,
      'max_sq_dist_nodes': sq_dist_node,
      'max_root_dist': max_root_dist,
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
