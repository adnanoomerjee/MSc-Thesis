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
      morphology: Literal['ant', 'humanoid'] = 'ant',
      goal_obs: Literal['concatenate', 'node'] = 'concatenate',
      position_goals: bool = True,
      velocity_goals: Literal[None, 'root', 'full'] = None, 
      goal_root_pos_range: jp.ndarray = jp.array([[-10,10], [-10,10], [-0.25, 0.45]]),
      goal_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi], [-jp.pi,jp.pi]]),
      goal_root_vel_range: jp.ndarray = jp.array([[-10,10], [-10,10], [-5, 5]]),
      goal_root_ang_range: jp.ndarray = jp.array([[-10,10], [-10,10], [-5, 5]]),
      goalsampler_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi/12], [-jp.pi,jp.pi]]),
      obs_mask: Optional[jp.ndarray] = None,
      distance_reward: Literal['difference', 'absolute'] = 'absolute',
      terminate_when_unhealthy=False,
      terminate_when_goal_reached=True,
      unhealthy_cost=0, # trial 0, -1.0, current best 0
      healthy_z_range=(0.255, 2.0),
      goal_distance_epsilon = 0.01, # current best 0.01
      reset_noise_scale=0.1,
      rot_dist=True, #trial, current best True
      backend='positional',
      architecture_name='Transformer',
      architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
      ctrl_cost=0.0, # trial 0, 0.5, current best 0
      reward_goal_reached=50, # trial 0, 50, 100, 500, current best 50
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
      # TODO: does the same actuator strength work as in spring
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

    # Reward attributes
    self.distance_reward = distance_reward

    # Termination attributes
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._terminate_when_goal_reached = terminate_when_goal_reached
    self.goal_distance_epsilon = goal_distance_epsilon
    self._healthy_z_range = healthy_z_range
    self.rot_dist = rot_dist

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

    self.goal_obs_width = goal_x_obs_width + goal_xd_obs_width
    self.action_mask = jp.concatenate([x_action_mask, xd_action_mask], axis=-1)

    self.limb_ranges = self._get_limb_ranges()
    self.max_goal_dist = self.limb_ranges['max_dist']

    # Goal sampling attributes
    if morphology == 'ant':
      self.end_effector_idx = [2,4,6,8]
      self.goal_z_cond = jp.array([0.078, 1.8])
      self.goal_polar_cond = jp.pi/12
      self.goal_contact_cond = 0.09
      self.air_probability = jp.array([0.9, 0.1])

    goalsampler_q_limit = jp.array(self.sys.dof.limit).T.at[0:3].set(self.goal_root_pos_range).at[3:6].set(goalsampler_root_rot_range)
    goalsampler_qd_limit = self.limb_ranges['goalsampler_qd_limit']
    self.goalsampler_limit = jp.concatenate([goalsampler_q_limit, goalsampler_qd_limit])

    # Training attributes
    max_actions_per_node = 24 if not self.velocity_goals else 48
    self.obs_mask = obs_mask
    self.non_actuator_nodes = None
    self.num_nodes = 5

    xd_action_mask = jp.array([
        [jp.ones(6)    , jp.zeros(6)   ],
        [jp.ones((4,6)), jp.ones((4,6))],
      ]
    )

    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name=architecture_name, **architecture_configs)
    num_attn_heads = self.network_architecture.configs['policy_num_heads'] if self.network_architecture.name=='Transformer' else 2
    self.max_actions_per_node = max_actions_per_node if self.network_architecture.name=='Transformer' else max_actions_per_node*5

    # Observation attributes
    self.state_obs_width = 13
    concat_obs_width = self.state_obs_width + self.goal_obs_width
    if concat_obs_width % num_attn_heads != 0:
        self.concat_obs_width = ((concat_obs_width // num_attn_heads) + 1) * num_attn_heads
    else:
        self.concat_obs_width = concat_obs_width

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

    logging.info('Environment initialised.')

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    low, hi = self.observer.goal_q_limit[:,0], self.observer.goal_q_limit[:,1]

    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    
    qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(q, qd)

    # Sample and set goal
    goal = self._sample_goal(rng3, pipeline_state) 

    goal_distance = self.goal_dist(pipeline_state, goal, frame = 'relative')/self.max_goal_dist #TODO
    goal_distance_root = self.root_goal_dist(pipeline_state, goal, rot_dist=self.rot_dist)/self.max_root_goal_dist #TODO
    goal_distance_world = self.goal_dist(pipeline_state, goal, frame = 'world')

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

    low_level_goal = self._get_low_level_goal(action) # TODO
    low_level_obs = self.low_level_env.get_obs(state, low_level_goal)

    rng = state.info['rng']
    goal = state.info['goal']

    rng, rng1 = jax.random.split(rng)

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

    # Compute state observation
    obs = self.get_obs(pipeline_state, goal) # TODO SLERP

    # Compute goal distance
    goal_distance = self.goal_dist(pipeline_state, goal, frame = 'relative')/self.max_goal_dist # TODO max_goal_dist
    
    # Compute rewards: R = ||s-g|| - ||s'-g||
    reward = self._get_intrinsic_reward(
        previous_state=prev_pipeline_state, 
        current_state=pipeline_state,
        goal=goal
    )

    # Check if goal reached
    goal_reached = jp.where(
      goal_distance < self.goal_distance_epsilon, x=1.0, y=0.0
    )

    if self._terminate_when_unhealthy:
      done = 0.0 + jp.logical_or(is_unhealthy, goal_reached)
    else:
      done = 0.0 + goal_reached

    state.metrics.update(
        reward=reward,
        goal_distance=goal_distance
    )

    state.info.update(
      rng=rng
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )


  def get_obs(self, state: base.State, goal: Goal) -> jp.ndarray:
    """Return observation input tensor"""

    def _get_obs_x(state: Union[base.State, Goal]):
        """Returns world root position (masking XY position) and limb position relative to parent"""
        x = world_to_egocentric(state.x, self.sys, mask_root = True)
        return self._average_x(x)
    
    def _get_obs_gx(goal: Goal, sx: base.State):
        """Returns world root position and egocentric limb position"""
        return goal.x_rel.vmap(in_axes = (0, 0)).to_local(sx) * self.goal_x_mask
    
    def _get_obs_xd(state: Union[base.State, Goal]):
        """Returns world root velocity and egocentric limb velocity"""
        xd = world_to_egocentric(state.xd, self.sys)
        return self._average_xd(xd)
      
    def _get_obs_gxd(goal: Goal, sxd: base.State):
        """Returns world root velocity and egocentric limb velocity"""
        return goal.xd_rel.__sub__(sxd) * self.goal_xd_mask
    
    sx = concatenate_attrs(_get_obs_x(state))
    sxd = _get_obs_xd(state)
    gx = concatenate_attrs(_get_obs_gx(goal, state)) if self.position_goals else jp.empty((self.num_links, 0))
    gxd = concatenate_attrs(_get_obs_gxd(goal, sxd)) if self.velocity_goals else jp.empty((self.num_links, 0))
    sxd = concatenate_attrs(sxd)

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
    if frame == 'world':
        state_x = state.x
        state_xd = state.xd 
        goal_x = goal.x_world
        goal_xd = goal.xd_world
    else:
        state_x = world_to_egocentric(state.x, self.sys)
        state_xd = world_to_egocentric(state.xd, self.sys)
        goal_x = goal.x_rel
        goal_xd = goal.xd_rel

    state_x = self._average_x(state_x)
    state_xd = self._average_xd(state_xd)
    
    dpos = state_x.pos - goal_x.pos
    drot = dist_quat(state_x.rot, goal_x.rot)
    dx = jp.concatenate([dpos, drot]) * self.goal_x_mask
    dxd = concatenate_attrs(state_xd.__sub__(goal_xd) * self.goal_xd_mask)
    s_minus_g = jp.concatenate([dx, dxd], axis=-1)
    return safe_norm(s_minus_g)
  

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
    x_world, xd_world = base.Transform(x.pos.at[:,2].set(z), x.rot), xd
    x_rel, xd_rel = self._average_x(world_to_egocentric(x_world, self.sys)), self._average_xd(world_to_egocentric(xd_world, self.sys))
    x_world, xd_world = self._average_x(x_world) * self.goal_x_mask, self._average_xd(xd_world) * self.goal_xd_mask
    x_rel, xd_rel = x_rel * self.goal_x_mask, xd_rel * self.goal_xd_mask   

    return Goal(g, q, qd, x_world, x_rel, xd_world, xd_rel)
  

  def _average_x(self, x: base.Transform):
    root_pos = x.pos[0]
    limbs_pos = jp.mean(x.pos[1:9].reshape(4, 2, -1), axis=-2)
    root_rot = x.rot[0] 
    limbs_rot = SLERP(x.rot[1:9].reshape(4, 2, -1)) # TODO
    pos = jp.concatenate([root_pos, limbs_pos], axis=-2)
    rot = jp.concatenate([root_rot, limbs_rot], axis=-2)
    return base.Transform(pos, rot)
    

  def _average_xd(self, xd: base.Motion):
    root_xd = xd.take(0)
    limbs_xd = xd.take(jp.array(range(1, 9)))
    limbs_xd = jax.tree_map(
        lambda m: jp.mean(m.reshape(4, 2, -1), axis=-2),
        limbs_xd
    )
    return root_xd.concatenate(limbs_xd)
    

