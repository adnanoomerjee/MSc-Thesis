## pylint:disable=g-multiple-import
"""Creates an environment for the lowest level of a hierarchical framework"""
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



class LowLevelEnv(PipelineEnv):

  def __init__(
    self,
    morphology: Literal['ant', 'humanoid'] = 'ant',
    goal_obs: Literal['concatenate', 'node'] = 'concatenate',
    position_goals: bool = True,
    velocity_goals: Literal[None, 'root', 'full'] = None, 
    goal_root_pos_masked = False,
    goal_root_pos_range: jp.ndarray = jp.array([[-3,3], [-3,3], [-0.25, 0.55]]),
    goal_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi], [-jp.pi,jp.pi]]),
    goal_root_vel_range: jp.ndarray = jp.array([[-10,10], [-10,10], [-5, 5]]),
    goal_root_ang_range: jp.ndarray = jp.array([[-5,5], [-5,5], [-10, 10]]),
    goalsampler_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi/12], [-jp.pi,jp.pi]]),
    obs_mask: Optional[jp.ndarray] = None,
    distance_reward: Literal['relative', 'absolute'] = 'relative',
    terminate_when_goal_reached=True,
    unhealthy_cost=-1.0, 
    air_probability=0.2, 
    goal_importance: Literal[None, 'continuous', 'discrete'] = None,
    reset_noise_scale=0.01,
    rot_dist=True,
    backend='positional',
    architecture_name='MLP',
    architecture_configs=DEFAULT_MLP_CONFIGS, # trial larger network
    ctrl_cost=0, 
    reward_goal_reached=0, 
    goal_distance_epsilon=0.1, 
    resample_probability=0,
    reset_interval=100,
    goal_dist: Literal['relative', 'absolute'] = 'relative',
    **kwargs
  ):

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')

    logging.info('Initialising environment...')
  
    path = epath.resource_path('hct') / f'envs/assets/{morphology}.xml'
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

    # Agent attributes
    self.dof = jp.array(self.sys.dof.limit).T.shape[0]
    self.num_links = sys.num_links()
    self.link_parents = sys.link_parents

    # Reward attributes
    self.distance_reward = distance_reward
    self.unhealthy_cost = unhealthy_cost
    self.ctrl_cost = ctrl_cost
    self.reward_goal_reached = reward_goal_reached

    # Termination attributes
    self._terminate_when_unhealthy = True 
    self._terminate_when_goal_reached = terminate_when_goal_reached
    self.goal_distance_epsilon = goal_distance_epsilon * 0.1 if goal_dist == 'relative' else goal_distance_epsilon
    self.rot_dist = rot_dist
    self.goal_dist = goal_dist

    # Reset attributes
    self._reset_noise_scale = reset_noise_scale

    # Goal attributes
    self.goal_nodes = True if goal_obs == 'node' else False
    self.position_goals = position_goals
    self.velocity_goals = False if velocity_goals is None else True
    self.root_velocity_goals = True if velocity_goals == 'root' else False
    self.full_velocity_goals = True if velocity_goals == 'full' else False
    self.goal_size = (self.dof*2,)
    self.goal_root_pos_range = goal_root_pos_range
    self.goal_root_rot_range = goal_root_rot_range
    self.goal_root_vel_range = goal_root_vel_range
    self.goal_root_ang_range = goal_root_ang_range
    self.goal_root_pos_masked  = goal_root_pos_masked 
    self.goal_importance = True if goal_importance is not None else False
    self.goal_importance_framework = goal_importance

    if self.position_goals and self.full_velocity_goals:
        self.goal_x_mask = 1
        self.goal_xd_mask = 1
        self.goal_obs_width = 13
    elif self.position_goals and self.root_velocity_goals:
        self.goal_x_mask = 1
        self.goal_xd_mask = jp.zeros((self.num_links,1)).at[0].set(1.0)
        self.goal_obs_width = 13
    elif self.position_goals:
        self.goal_x_mask = 1
        self.goal_xd_mask = 0
        self.goal_obs_width = 7
    elif self.full_velocity_goals:
        self.goal_x_mask = 0
        self.goal_xd_mask = 1
        self.goal_obs_width = 6
    else:
        assert self.position_goals, "Cannot only specify root_velocity_goals"
    
    self.goal_obs_width += self.goal_importance

    if goal_root_pos_masked:
        self.goal_x_mask = jp.ones((self.num_links,1)).at[0,0].set(0)

    # Training attributes
    self.obs_mask = obs_mask
    self.non_actuator_nodes = 0 if not self.goal_nodes else jp.array([0] + [i for i in range(9, 18)])
    self.action_mask = None
    self.num_nodes = 9

    self.limb_ranges = self._get_limb_ranges()
    self.max_goal_dist = self.limb_ranges['max_dist']
    self.max_root_goal_dist = self.limb_ranges['max_root_dist']
    self.pos_range = self.limb_ranges['pos_range']
    self.rot_range = self.limb_ranges['rot_range']
    self.vel_range_range = self.limb_ranges['vel_range']
    self.ang_range = self.limb_ranges['ang_range']
    self.max_sq_dist_nodes = self.limb_ranges['max_sq_dist_nodes']
    self.minmax = self.limb_ranges['minmax']

    # Goal sampling attributes
    self.goal_z_cond = jp.array([0.078, 1.6]) if backend == 'generalized' else jp.array([0.08, 1.6])
    self.goal_polar_cond = jp.pi/12
    self.goal_contact_cond = 0.09
    self.air_probability = jp.array([1-air_probability, air_probability])

    goalsampler_q_limit = jp.array(self.sys.dof.limit).T.at[0:3].set(self.goal_root_pos_range).at[3:6].set(goalsampler_root_rot_range)
    goalsampler_qd_limit = self.limb_ranges['goalsampler_qd_limit']
    self.goalsampler_limit = jp.concatenate([goalsampler_q_limit, goalsampler_qd_limit])

    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name=architecture_name, **architecture_configs)
    num_attn_heads = self.network_architecture.configs['policy_num_heads'] if self.network_architecture.name=='Transformer' else 2
    self.max_actions_per_node = 1 if self.network_architecture.name=='Transformer' else 8

    # Observation attributes
    self.root_mask = base.Transform(jp.array([0.0, 0.0, 1.0]), jp.array([1.0, 1.0, 1.0, 1.0]))
    self.state_obs_width = 13
    concat_obs_width = self.state_obs_width + self.goal_obs_width
    if concat_obs_width % num_attn_heads != 0:
        self.concat_obs_width = ((concat_obs_width // num_attn_heads) + 1) * num_attn_heads
    else:
        self.concat_obs_width = concat_obs_width

    self.action_repeat = 1
    self.horizon = 20
    self.episode_length = self.horizon * reset_interval
    self.action_shape = (8, 1)
    self.resample_probability=resample_probability
    self.reset_interval=reset_interval

    self.goal_ranges = jp.concatenate(
        [
          jp.array(self.pos_range),
          jp.squeeze(jp.array(self.rot_range)),
          jp.array(self.vel_range_range),
          jp.array(self.ang_range)
        ],
        axis=-1
      )

    logging.info('Environment initialised.')


  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    rng, rng1, rng2 = jax.random.split(rng, 3)

    pipeline_state = self._sample_state(rng1)

    # Sample and set goal
    goal = self._sample_goal(rng2, pipeline_state)

    # Get observation
    obs, goal_dist = self.get_obs(pipeline_state, goal, goal_dist=True)
    #goal_dist_normalised = goal_dist/self.max_goal_dist
    
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
      prev_goal_dist <= self.goal_distance_epsilon * jp.sum(goal.importance), x=1.0, y=0.0
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
    pipeline_state = jax.lax.cond(done, rng2, lambda x: self._sample_state(x), (pipeline_state, action), lambda x: self.pipeline_step(x[0], x[1]))
    goal = jax.lax.cond(resample_goal, self._sample_goal, lambda x, y: goal, rng3, pipeline_state)
    
    # Compute state observation
    obs, goal_dist = self.get_obs(pipeline_state, goal, goal_dist=True)

    # Compute rewards: 
    if self.distance_reward == 'absolute':
      reward = -goal_dist
    else:
      reward = (prev_goal_dist - goal_dist)/self.dt # - 0.1 * jp.sum(jp.square(action))

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

  def get_obs(self, state: base.State, goal: Goal, goal_dist=False) -> jp.ndarray:
    """
    Processes a state and goal into observation format

    Args:
        state: dynamic State object, changing every step
        goal: Goal object, containing egocentric Transform goals
          and/or world frame velocity goals

    Returns:
        obs: (num_links, 13 + goal_size) array containing goal observations:
    """

    def _mask_root(x):
      root_x = x.take(0)
      root_x = mul(root_x, self.root_mask)
      return x.index_set(0, root_x)
    
    def _get_state_obs(state: Union[base.State, Goal]):
      """Returns world root position (masking XY position) and limb position relative to parent"""
      return self._world_to_relative(state)
    
    def _get_goal_obs(goal: Goal, sx: base.Transform, sxd: base.Motion):
      """Returns world root pos/vel and relative limb pos/vel"""
      if self.position_goals:
        gx = goal.x_rel.vmap(in_axes = (0, 0)).to_local(sx) * self.goal_x_mask
        rot = gx.rot.at[1:].set(jp.array([1, 0, 0, 0]))
        gx = base.Transform(gx.pos, rot)
        gx = concatenate_attrs(gx)
      else:
        gx = jp.empty((self.num_links, 0))
      if self.velocity_goals:
        gxd = jax.tree_map(lambda x: jax.vmap(inv_rotate)(x, sx.rot), goal.xd_rel.__sub__(sxd)) * self.goal_xd_mask 
        gxd = concatenate_attrs(gxd)
      else:
        gxd = jp.empty((self.num_links, 0))
      return gx, gxd
          
    state_x, state_xd =  _get_state_obs(state)
    gx, gxd =  _get_goal_obs(goal, state_x, state_xd)

    sx, sxd = concatenate_attrs(_mask_root(state_x)), concatenate_attrs(state_xd)

    importance = goal.importance if self.goal_importance else jp.empty((self.num_links, 0))

    s_obs = jp.concatenate([sx, sxd], axis = -1)
    g_obs = jp.concatenate([gx, gxd, importance], axis = -1)

    if self.goal_nodes:
      g_obs = pad(g_obs, self.state_obs_width)
      s_obs = jp.concatenate([s_obs, jp.zeros((self.num_links,1))], axis = -1)
      g_obs = jp.concatenate([s_obs, jp.ones((self.num_links,1))], axis = -1)
      obs = jp.concatenate([s_obs, g_obs], axis = -2)
    else:
      obs = jp.concatenate([s_obs, g_obs], axis=-1)
      obs = pad(obs, self.concat_obs_width)

    if self.network_architecture.name == 'MLP':
      obs = obs.reshape(*obs.shape[:-2], -1)

    if goal_dist:
      if self.goal_dist == 'relative':
        goal_dist = self._dist(state_x, state_xd, goal.x_rel, goal.xd_rel, importance=goal.importance)
      else:
        goal_dist = self._dist(state.x, state.xd, goal.x, goal.xd, importance=goal.importance)
    else:
      goal_dist = None

    return obs, goal_dist


  
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


  def _get_limb_ranges(self):

    filepath = '/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/envs/ranges/'

    env = 'Low level'

    max_ja = [ 0.80290073, 1.3839247, 0.80290073, 1.3839247, 0.80290073, 1.3839247, 0.80290073, 1.3839247 ]
    max_jv = [22.484863, 17.487995, 22.484863, 18.571373, 22.484863, 22.484863, 22.484863, 22.484863]
    min_ja = [-0.80630934, 0, -0.80630934, 0., -0.80630934, 0. -0.80630934, 0.]
    min_jv = [-22.484863, -17.487995, -22.484863, -18.571373, -22.484863, -22.484863, -22.484863, -22.484863]

    if self.root_velocity_goals:
      velocity_goals = 'root'
    elif self.full_velocity_goals:
      velocity_goals = 'full'
    else:
      velocity_goals = 'False'
    
    position_goals = str(self.position_goals)
    rot_dist = str(self.rot_dist)

    variant = f'{env}, {position_goals}, {velocity_goals}, {rot_dist}, {self.goal_dist}'

    #if self.goal_root_pos_masked :
    #  variant = f'{env}, {position_goals}, {velocity_goals}, {rot_dist}, {self.goal_root_pos_masked}'
    
    filename = f'{filepath}/{variant}'
    
    if os.path.isfile(filename):
      return model.load(filename)
    
    quaternion_to_spherical_vmap = jax.vmap(quaternion_to_spherical, in_axes=0)

    test_rollout, _ = AntTest().test_rollout()

    rollout_rel = [self._world_to_relative(state) for state in test_rollout]
    rollout_rel_x = [t[0] for t in rollout_rel]
    rollout_rel_xd = [t[1] for t in rollout_rel]

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

    minmax['pos'] = jp.abs(x_min.pos - x_max.pos) #if self.goal_dist == 'relative' else 6
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

    model.save(filename, return_dict)
    return return_dict
  
  def _world_to_relative(self, state: base.State):
    return world_to_relative(state, self)
  
  def _dist(self, state_x, state_xd, goal_x, goal_xd, importance):
    return dist(self, state_x, state_xd, goal_x, goal_xd, importance=importance)[0]
  

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
    rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
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
    x_rel, xd_rel = self._world_to_relative(goal_state) 

    if self.goal_importance:
      if self.goal_importance_framework == 'continuous':
        importance = jax.random.uniform(rng4, shape=(self.num_nodes, 1), minval=0, maxval=1)# * choice + jp.zeros((9,1)).at[0,0].set(1.) * (1-choice) #(jax.nn.sigmoid(jax.random.uniform(rng4, shape=(self.num_nodes, 1), minval = -40, maxval = 10))) #* choice + jp.zeros((9,1)).at[0,0].set(1.) * (1-choice)
      else:
        importance = jax.random.choice(rng4, shape=(self.num_nodes, 1), a=jp.array([0.0, 1.0]))# * choice + jp.zeros((9,1)).at[0,0].set(1.) * (1-choice) #(jax.nn.sigmoid(jax.random.uniform(rng4, shape=(self.num_nodes, 1), minval = -40, maxval = 10))) #* choice + jp.zeros((9,1)).at[0,0].set(1.) * (1-choice)
    else:
      importance = 1

    return Goal(q, qd, x, xd, x_rel, xd_rel, importance=importance)
  
    '''
      def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    if action.shape[-1] == 1:
      action = jp.squeeze(action, axis=-1) 

    goal = state.info['goal']
    goal_count = state.info['goal_count']
    rng = state.info['rng']
    prev_goal_dist = state.metrics['goal_dist']
    sub_episode_step = state.info['sub_episode_step']

    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    pipeline_state = state.pipeline_state

    # Check if goal reached
    goal_reached = jp.where(
      prev_goal_dist <= self.goal_distance_epsilon, x=1.0, y=0.0
    )


    # Check if unhealthy
    is_unhealthy = jp.where(rotate(jp.array([0, 0, 1]), pipeline_state.x.rot[0])[-1] < 0, x=1.0, y=0.0)

    random_resample_goal = jax.random.choice(rng1, jp.array([0, 1]), p=jp.array([1-self.resample_probability, self.resample_probability]))
    
    done = jp.where(
      is_unhealthy, x=1.0, y=jp.where(goal_count >= self.reset_interval, x=1.0, y=0.0))
    
    resample_goal = jp.where(
      random_resample_goal > 0.5, x=1.0, y=jp.where(jp.logical_or(done, goal_reached), 
                                                    x=1.0, y=jp.where(sub_episode_step >= self.horizon, x=1.0, y=0.0)))
    
    sub_episode_step = jp.where(resample_goal, x=0.0, y=sub_episode_step + 1)
    goal_count = jp.where(done, x=0.0, y=goal_count + resample_goal)
    goals_reached = (state.metrics['goals_reached'] + goal_reached)/(goal_count+1)

    # Take action
    pipeline_state = jax.lax.cond(done, rng2, lambda x: self._sample_state(x), (pipeline_state, action), lambda x: self.pipeline_step(x[0], x[1]))
    goal = jax.lax.cond(resample_goal, self._sample_goal, lambda x, y: goal, rng3, pipeline_state)
    
    # Compute state observation
    obs, goal_dist = self.get_obs(pipeline_state, goal, goal_dist=True)

    # Compute rewards: 
    if self.distance_reward == 'absolute':
      reward = -goal_dist
    else:
      reward = (prev_goal_dist - goal_dist)/self.dt

    state.info.update(
      goal=goal,
      rng=rng,
      goal_count=goal_count,
      sub_episode_step=sub_episode_step
    )

    state.metrics.update(
      reward=reward,
      goal_dist=goal_dist,
      is_unhealthy=1.0*is_unhealthy,
      goals_reached=goals_reached
    )

    return state.replace(
      pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
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
    rng, rng1, rng2, rng4 = jax.random.split(rng, 4)
    g = jax.random.uniform(
      rng, 
      shape=(self.num_links, 12), 
      minval=self.goal_ranges[0], 
      maxval=self.goal_ranges[1]
    )

    pos_goals = g[:, 0:3]
    rot_goals = jax.vmap(spherical_to_quaternion)(g[:, 3:6])
    vel_goals = g[:, 6:9]
    ang_goals = g[:, 9:12]
    
    x_rel = base.Transform(pos_goals, rot_goals)
    xd_rel = base.Motion(vel_goals, ang_goals)

    x_rel = base.Transform(x_rel.pos + state.x.pos, x_rel.rot)

    rel_state = base.State(None, None, x=x_rel, xd=xd_rel, contact=None)

    x, xd = relative_to_world(rel_state, self)

    if self.goal_importance:
      importance = jax.nn.sigmoid(jax.random.uniform(rng4, shape=(self.num_nodes, 1), minval = -40, maxval = 10))
    else:
      importance = None

    return Goal(None, None, x, xd, x_rel, xd_rel, importance=importance)
  
    












    def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    rng, rng1, rng2 = jax.random.split(rng, 3)

    pipeline_state = self._sample_state(rng1)

    # Sample and set goal
    goal = self._sample_goal(rng2, pipeline_state)

    # Get observation
    obs, goal_dist = self.get_obs(pipeline_state, goal, goal_dist=True)
    #goal_dist_normalised = goal_dist/self.max_goal_dist
    
    # Set metrics
    reward, done = jp.zeros(2)

    if self.distance_reward == 'absolute':
      reward= -goal_dist

    metrics = {
      'reward': reward,
      'goal_dist':goal_dist,
      'is_unhealthy': 0.0,
      'goals_reached': 0.0
    }

    info = {
      'goal': goal,
      'goal_count': 0.0,
      'rng': rng
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

    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    pipeline_state = state.pipeline_state

    # Check if goal reached
    goal_reached = jp.where(
      prev_goal_dist <= self.goal_distance_epsilon, x=1.0, y=0.0
    )

    # Check if unhealthy
    is_unhealthy = jp.where(rotate(jp.array([0, 0, 1]), pipeline_state.x.rot[0])[-1] < 0, x=1.0, y=0.0)

    random_resample_goal = jax.random.choice(rng1, jp.array([0, 1]), p=jp.array([1-self.resample_probability, self.resample_probability]))

    done = jp.where(
      random_resample_goal > 0.5, x=1.0, y=jp.where(jp.logical_or(is_unhealthy, goal_reached), x=1.0, y=0.0))
    
    goal_count = goal_count + done 

    reset_state = jp.where(
      is_unhealthy, x=1.0, y=jp.where(goal_count >= self.reset_interval, x=1.0, y=0.0))

    goal_count = jp.where(reset_state, x=0.0, y=goal_count)

    # Take action
    pipeline_state = jax.lax.cond(reset_state, rng2, lambda x: self._sample_state(x), (pipeline_state, action), lambda x: self.pipeline_step(x[0], x[1]))
    goal = jax.lax.cond(done, self._sample_goal, lambda x, y: goal, rng3, pipeline_state)
    
    # Compute state observation
    obs, goal_dist = self.get_obs(pipeline_state, goal, goal_dist=True)

    state.info['first_pipeline_state'] = state.pipeline_state
    state.info['first_obs'] = state.obs

    # Compute rewards: 
    if self.distance_reward == 'absolute':
      reward = -goal_dist
    else:
      reward = (prev_goal_dist - goal_dist)

    #reward = (1 - is_unhealthy) * reward - is_unhealthy

    state.info.update(
      goal=goal,
      rng=rng,
      goal_count=goal_count,
    )

    state.metrics.update(
      reward=reward,
      goal_dist=goal_dist,
      is_unhealthy=1.0*is_unhealthy,
      goals_reached=1.0*goal_reached
    )

    return state.replace(
      pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
    '''