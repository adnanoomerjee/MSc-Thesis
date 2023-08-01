import sys
import inspect
import os

from hct.envs.goal import Goal
from hct.envs.ant_test import AntTest
from hct.envs.tools import *
from hct.training.configs import NetworkArchitecture, SMALL_TRANSFORMER_CONFIGS, DEFAULT_MLP_CONFIGS
from hct.io import model

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

  # pyformat: disable
  """
  ### Description

  This environment is based on the environment introduced by Schulman, Moritz,
  Levine, Jordan and Abbeel in
  ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).

  The ant is a 3D robot consisting of one torso (free rotational body) with four
  legs attached to it with each leg having two links.

  The goal is to coordinate the four legs to move in the forward (right)
  direction by applying torques on the eight hinges connecting the two links of
  each leg and the torso (nine parts and eight hinges).

  ### Action Space

  The agent take a 8-element vector for actions.

  The action space is a continuous `(action, action, action, action, action,
  action, action, action)` all in `[-1, 1]`, where `action` represents the
  numerical torques applied at the hinge joints.

  | Num | Action                                                             | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit         |
  |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
  | 0   | Torque applied on the rotor between the torso and front left hip   | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
  | 1   | Torque applied on the rotor between the front left two links       | -1          | 1           | ankle_1 (front_left_leg)         | hinge | torque (N m) |
  | 2   | Torque applied on the rotor between the torso and front right hip  | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
  | 3   | Torque applied on the rotor between the front right two links      | -1          | 1           | ankle_2 (front_right_leg)        | hinge | torque (N m) |
  | 4   | Torque applied on the rotor between the torso and back left hip    | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
  | 5   | Torque applied on the rotor between the back left two links        | -1          | 1           | ankle_3 (back_leg)               | hinge | torque (N m) |
  | 6   | Torque applied on the rotor between the torso and back right hip   | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
  | 7   | Torque applied on the rotor between the back right two links       | -1          | 1           | ankle_4 (right_back_leg)         | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  ant, followed by the velocities of those individual parts (their derivatives)
  with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(27,)` where the elements correspond to the following:

  | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
  |-----|--------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the torso (centre)                           | -Inf | Inf | torso                            | free  | position (m)             |
  | 1   | w-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 2   | x-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 3   | y-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 4   | z-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 5   | angle between torso and first link on front left             | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
  | 6   | angle between the two links on the front left                | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
  | 7   | angle between torso and first link on front right            | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
  | 8   | angle between the two links on the front right               | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
  | 9   | angle between torso and first link on back left              | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
  | 10  | angle between the two links on the back left                 | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
  | 11  | angle between torso and first link on back right             | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
  | 12  | angle between the two links on the back right                | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |
  | 13  | x-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 14  | y-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 15  | z-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 16  | x-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 17  | y-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 18  | z-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 19  | angular velocity of angle between torso and front left link  | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
  | 20  | angular velocity of the angle between front left links       | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
  | 21  | angular velocity of angle between torso and front right link | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
  | 22  | angular velocity of the angle between front right links      | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
  | 23  | angular velocity of angle between torso and back left link   | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
  | 24  | angular velocity of the angle between back left links        | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
  | 25  | angular velocity of angle between torso and back right link  | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
  | 26  | angular velocity of the angle between back right links       | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |

  The (x,y,z) coordinates are translational DOFs while the orientations are
  rotational DOFs expressed as quaternions.

  ### Rewards

  The reward consists of three parts:

  - *reward_survive*: Every timestep that the ant is alive, it gets a reward of
    1.
  - *reward_forward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
    time between actions - the default *dt = 0.05*. This reward would be
    positive if the ant moves forward (right) desired.
  - *reward_ctrl*: A negative reward for penalising the ant if it takes actions
    that are too large. It is measured as *coefficient **x**
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.5.
  - *contact_cost*: A negative reward for penalising the ant if the external
    contact force is too large. It is calculated *0.5 * 0.001 *
    sum(clip(external contact force to [-1,1])<sup>2</sup>)*.

  ### Starting State

  All observations start in state (0.0, 0.0,  0.75, 1.0, 0.0  ... 0.0) with a
  uniform noise in the range of [-0.1, 0.1] added to the positional values and
  standard normal noise with 0 mean and 0.1 standard deviation added to the
  velocity values for stochasticity.

  Note that the initial z coordinate is intentionally selected to be slightly
  high, thereby indicating a standing up ant. The initial orientation is
  designed to make it face forward as well.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps
  2. The y-orientation (index 2) in the state is **not** in the range
     `[0.2, 1.0]`
  """
  # pyformat: enable


  def __init__(
      self,
      low_level_env: Env,
      low_level_make_inference_fn: Callable,
      low_level_params: types.PolicyParams,
      low_level_network: FeedForwardNetwork,
      num_nodes=5,
      action_repeat=5,
      goal_distance_epsilon = 0.01, # current best 0.01
      rot_dist=True, #trial, current best True
      architecture_name='Transformer',
      architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
      **kwargs
  ):
    
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')

    logging.info('Initialising environment...')

    morphology = low_level_env.morphology
  
    path = epath.resource_path('hct') / f'envs/assets/{morphology}.xml'
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
    self.link_parents = (-1, 0, 0, 0, 0)

    # Reward attributes
    self.distance_reward = low_level_env.distance_reward

    # Termination attributes
    self._terminate_when_unhealthy = low_level_env.terminate_when_unhealthy
    self._terminate_when_goal_reached = low_level_env.terminate_when_goal_reached
    self.goal_distance_epsilon = goal_distance_epsilon
    self._healthy_z_range = low_level_env.healthy_z_range
    self.rot_dist = rot_dist

    # Reset attributes
    self._reset_noise_scale = low_level_env.reset_noise_scale

    # Training attributes
    self.obs_mask = low_level_env.obs_mask
    self.non_actuator_nodes = None
    self.num_nodes = num_nodes

    # Goal attributes
    self.goal_nodes = low_level_env.goal_nodes
    self.position_goals = low_level_env.position_goals
    self.velocity_goals = low_level_env.position_goals
    self.root_velocity_goals = low_level_env.root_velocity_goals
    self.full_velocity_goals = low_level_env.full_velocity_goals
    self.goal_size = low_level_env.goal_size
    self.goal_root_pos_range = low_level_env.goal_root_pos_range
    self.goal_root_rot_range = low_level_env.goal_root_rot_range
    self.goal_root_vel_range = low_level_env.goal_root_vel_range
    self.goal_root_ang_range = low_level_env.goal_root_ang_range
    self.goal_x_mask = low_level_env.goal_x_mask
    self.goal_xd_mask = low_level_env.goal_xd_mask

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
      self.action_mask = jp.ones((num_nodes,24)).at[0,12:].set(0)
      max_actions_per_node = 24
    elif self.position_goals and self.root_velocity_goals:
      self.action_mask = jp.ones((num_nodes,18)).at[1:,6:12].set(0).at[0,12:18].set(0)
      max_actions_per_node = 18
    else:
      self.action_mask = jp.ones((num_nodes,12)).at[0,6:].set(0)
      max_actions_per_node = 12

    self.low_level_goal_ranges = jp.concatenate(
        [
          self.low_level_env.pos_range,
          self.low_level_env.rot_range,
          self.low_level_env.vel_range,
          self.low_level_env.ang_range
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


    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name=architecture_name, **architecture_configs)
    num_attn_heads = self.network_architecture.configs['policy_num_heads'] if self.network_architecture.name=='Transformer' else 2
    self.max_actions_per_node = max_actions_per_node if self.network_architecture.name=='Transformer' else max_actions_per_node * num_nodes

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
    goal_distance, goal_distance_root = self.goal_dist(pipeline_state, goal, frame='relative', root_dist=True)
    goal_distance = goal_distance/self.max_goal_dist
    goal_distance_world, _ = self.goal_dist(pipeline_state, goal, frame='world', root_dist=False)

    # Get observation
    obs = self.get_obs(pipeline_state, goal)
    
    # Set metrics
    reward, done = jp.zeros(2)

    if self.distance_reward == 'absolute':
      reward= -goal_distance_world

    metrics = {
      'reward': reward,
      'goal_distance_relative_frame_normalised': goal_distance,
      'goal_distance_world_frame': goal_distance_world,
      'goal_distance_root_normalised': goal_distance_root,
      'is_unhealthy': 0.0
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
    prev_goal_distance_world = state.metrics['goal_distance_world_frame']

    rng, rng1 = jax.random.split(rng)

    low_level_goal = self._get_low_level_goal(action) 
    low_level_obs = self.low_level_env.get_obs(state, low_level_goal)
    action, _ = self.low_level_policy(low_level_obs, rng1) 

    # Take action
    prev_pipeline_state = state.pipeline_state
    pipeline_state = self.pipeline_step(prev_pipeline_state, action) 

    # Check if unhealthy
    min_z, max_z = self._healthy_z_range
    is_unhealthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, x=1.0, y=0.0)
    is_unhealthy = jp.where(
        pipeline_state.x.pos[0, 2] > max_z, x=1.0, y=is_unhealthy
    )

    # Compute goal distance
    goal_distance, goal_distance_root = self.goal_dist(pipeline_state, goal, frame='relative', root_dist=True)
    goal_distance = goal_distance/self.max_goal_dist
    goal_distance_world, _ = self.goal_dist(pipeline_state, goal, frame='world', root_dist=False)

        # Check if goal reached
    goal_reached = jp.where(
      goal_distance < self.goal_distance_epsilon, x=1.0, y=0.0
    )

    # Compute rewards: 
    if self.distance_reward == 'absolute':
      reward = -goal_distance_world
    else:
      reward = prev_goal_distance_world - goal_distance_world

    reward += -self.ctrl_cost * jp.sum(jp.square(action)) + self.reward_goal_reached * goal_reached - is_unhealthy * self.unhealthy_cost

    # Compute state observation
    obs = self.get_obs(pipeline_state, goal) 

    if self._terminate_when_unhealthy:
      done = 0.0 + jp.logical_or(is_unhealthy, goal_reached)
    else:
      done = 0.0 + goal_reached

    state.metrics.update(
      reward=reward,
      goal_distance_relative_frame_normalised=goal_distance,
      goal_distance_world_frame=goal_distance_world,
      goal_distance_root_normalised=goal_distance_root,
      is_unhealthy=1.0*is_unhealthy
    )

    state.info.update(
      rng=rng
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
  
  def _get_low_level_goal(self, g):
    
    if self.position_goals and not self.velocity_goals:
      g = jp.insert(g, [6, 18], jp.zeros((self.num_nodes, 6)))
    elif self.velocity_goals and not self.position_goals:
      g = jp.insert(g, [0, 12], jp.zeros((self.num_nodes, 6)))
    elif self.root_velocity_goals:
      g = jp.insert(g, 18, jp.zeros((self.num_nodes, 6)))

    root_goals = g[0:12]
    upper_limb_goals = g[:, :24//2].reshape(-1, 1, 24//2)
    lower_limb_goals = g[:, 24[-1]//2:].reshape(-1, 1, 24[-1]//2)
    limb_goals = jp.concatenate([upper_limb_goals, lower_limb_goals], axis=-1)

    goals = jp.concatenate([root_goals, limb_goals])
    goals_unnormalised = unnormalize_to_range(
      goals,
      self.low_level_goal_ranges[0],
      self.low_level_goal_ranges[1],
      -1,
      1
    )

    pos_goals = goals_unnormalised[:, 0:3]
    rot_goals = goals_unnormalised[:, 3:6]
    vel_goals = goals_unnormalised[:, 6:9]
    ang_goals = goals_unnormalised[:, 9:12]

    x_rel = base.Transform(pos_goals, rot_goals)
    xd_rel = base.Motion(vel_goals, ang_goals)

    return Goal(g, None, None, None, None, x_rel, xd_rel)


  def get_obs(self, state: base.State, goal: Goal) -> jp.ndarray:
    """Return observation input tensor"""

    def _mask_root(x):
      root_x = x.take(0)
      root_x = mul(root_x, self.root_mask)
      return x.index_set(0, root_x)
    
    def _get_state_obs(state: Union[base.State, Goal]):
        """Returns world root position (masking XY position) and limb position relative to parent"""
        state = self._average_limbs(state)
        return self._world_to_relative(state)
    
    def _get_goal_obs(goal: Goal, sx: base.Transform, sxd: base.Motion):
      """Returns world root pos/vel and relative limb pos/vel"""
      if self.position_goals:
        gx = goal.x_rel.vmap(in_axes = (0, 0)).to_local(sx) * self.goal_x_mask
        gx = concatenate_attrs(gx)
      else:
        gx = jp.empty((self.num_links, 0))
      if self.velocity_goals:
        gxd = jax.tree_map(lambda x: jax.vmap(inv_rotate)(x, sx.rot), goal.xd_rel.__sub__(sxd)) * self.goal_xd_mask 
        gxd = concatenate_attrs(gxd)
      else:
        gxd = jp.empty((self.num_links, 0))
      return gx, gxd
    
    sx, sxd =  _get_state_obs(state)
    gx, gxd =  _get_goal_obs(goal, sx, sxd)
    
    sx, sxd = concatenate_attrs(_mask_root(sx)), concatenate_attrs(sxd)

    s_obs = jp.concatenate([sx, sxd], axis = -1)
    g_obs = jp.concatenate([gx, gxd], axis = -1)

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
    
    return obs
  

  def goal_dist(self, state: base.State, goal: Goal, frame: Literal['world', 'relative']):
    """
    Computes distance d(s,g) between state and goal in world frame, 
    accounting for quaternion double cover.

    dist(s,g) = ||s-g||
    """

    state = self._average_limbs(state)

    if frame == 'world':
        state_x = state.x
        state_xd = state.xd 
        goal_x = goal.x_world
        goal_xd = goal.xd_world
    else:
        state_x, state_xd = self._world_to_relative(state)
        goal_x = goal.x_rel
        goal_xd = goal.xd_rel

    dist, root_dist = dist(self, state_x, state_xd, goal_x, goal_xd, root_dist)
    return dist, root_dist
  

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
    goal_state = self._average_limbs(base.State(q, qd, x, xd, None))

    x_rel, xd_rel = self._world_to_relative(goal_state)   

    return Goal(g, q, qd, x, xd, x_rel, xd_rel)
  

  def _average_limbs(self, state):

    slerp = jax.vmap(slerp)

    def _average_x(x: base.Transform):
      root_pos = x.pos[0]
      limbs_pos = jp.mean(x.pos[1:].reshape(4, 2, -1), axis=-2)
      root_rot = x.rot[0] 
      limbs_rot = slerp(x.rot[jp.array(1,3,5,7)], x.rot[jp.array(2,4,6,8)]) 
      pos = jp.concatenate([root_pos, limbs_pos], axis=-2)
      rot = jp.concatenate([root_rot, limbs_rot], axis=-2)
      return base.Transform(pos, rot)
      
    def _average_xd(xd: base.Motion):
      root_xd = xd.take(0)
      limbs_xd = xd.take(jp.array(range(1, 9)))
      limbs_xd = jax.tree_map(
          lambda m: jp.mean(m.reshape(4, 2, -1), axis=-2),
          limbs_xd
      )
      return root_xd.concatenate(limbs_xd)
    
    x, xd = _average_x(state.x), _average_xd(state.xd)

    return base.State(
        q=state.q, 
        qd=state.qd,
        x=x,
        xd=xd,
        contact=None
        )
  

  def _world_to_relative(self, state: base.State):
    return world_to_relative(state, self)


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

    variant = f'{env}, {position_goals}, {velocity_goals}, {rot_dist}'
    
    filename = f'{filepath}/{variant}'
    
    if os.path.isfile(filename):
      return model.load(filename)

    quaternion_to_spherical_vmap = jax.vmap(quaternion_to_spherical, in_axes=0)

    test_rollout, _ = AntTest().test_rollout()

    rollout_rel = [self._world_to_relative(self._average_limbs(state)) for state in test_rollout]
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
    x_min = rollout_rel_x[7*30-1].index_set(0, base.Transform(self.goal_root_pos_range[:,0], jp.array([1,0,0,0])))
    x_max = rollout_rel_x[15*30-1].index_set(0, base.Transform(self.goal_root_pos_range[:,1], jp.array([0,1,0,0])))
    xd_min = base.Motion(vel_ranges[0], ang_ranges[0])
    xd_max = base.Motion(vel_ranges[1], ang_ranges[1])


    max_dist, max_root_dist = dist(self, x_min, xd_min, x_max, xd_max, root_dist=True)

    return_dict = {
      'max_dist': max_dist,
      'max_root_dist': max_root_dist,
      'pos_range': pos_ranges, 
      'rot_range': rot_ranges, 
      'vel_range': vel_ranges, 
      'ang_range': ang_ranges, 
      'goalsampler_qd_limit': goalsampler_qd_limit,
    }

    model.save(filename, return_dict)
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
