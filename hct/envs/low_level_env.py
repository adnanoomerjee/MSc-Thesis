# pylint:disable=g-multiple-import
"""Creates an environment for the lowest level of a hierarchical framework"""
import sys
sys.path.insert(0, "/nfs/nhome/live/aoomerjee/MSc-Thesis/")

from hct.envs.goal import Goal
from hct.envs.tools import *

from brax import base, generalized
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward
from brax.geometry import contact

from etils import epath

import jax
from jax import numpy as jp

from typing import Optional, Literal, Tuple


class LowLevelEnv(PipelineEnv):

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

  # TODO account for humanoid env, get ranges for relative pos/vel, compute max goal dist, configure metrics

  '''
  Test:
    pos goals
    pos + rootvel goals
    pos + full vel goals
    full vel goals

    goal concat vs goal nodes (goal distance)

    absolute vs difference reward (goal distance)
  '''

  def __init__(
      self,
      env_name: Literal['ant', 'humanoid'] = 'ant',
      goal_obs: Literal['concatenate', 'node'] = 'concatenate',
      position_goals: bool = True,
      velocity_goals: Literal[None, 'root', 'full'] = None, 
      goal_x_range: Tuple[base.Transform, base.Transform] = None,
      goal_xd_range: Tuple[base.Motion, base.Motion] = None,
      goalsampler_root_pos_range: jp.ndarray = jp.array([[-3,3], [-3,3], [-0.25, 0.6]]),
      goalsampler_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi], [-jp.pi, jp.pi]]),
      goalsampler_qd_limit: Optional[jp.ndarray] = None,
      max_goal_dist: Optional[jp.ndarray] = None,
      obs_mask: Optional[jp.ndarray] = None,
      action_mask: Optional[jp.ndarray] = None,
      distance_reward: Literal['difference', 'absolute'] = 'difference',
      unhealthy_cost=1.0,
      terminate_when_unhealthy=True,
      terminate_when_goal_reached=True,
      healthy_z_range=(0.2, 2.0),
      goal_distance_epsilon = 0.01,
      reset_noise_scale=0.1,
      backend='generalized',
      **kwargs,
  ):
    
    path = epath.resource_path('hct') / f'envs/assets/{env_name}.xml'
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

    # Reward attributes
    self.distance_reward = distance_reward
    self.unhealthy_cost = unhealthy_cost

    # Termination attributes
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._terminate_when_goal_reached = terminate_when_goal_reached
    self.goal_distance_epsilon = goal_distance_epsilon
    self._healthy_z_range = healthy_z_range

    # Reset attributes
    self._reset_noise_scale = reset_noise_scale

    # State attributes
    self.state_obs_width = 13

    # Goal attributes
    self.goal_nodes = True if goal_obs == 'node' else False
    self.position_goals = position_goals
    self.velocity_goals = False if velocity_goals is None else True
    self.root_velocity_goals = True if velocity_goals == 'root' else False
    self.full_velocity_goals = True if velocity_goals == 'full' else False
    self.goal_size = (self.dof*2,)
    assert self.position_goals or self.velocity_goals, "Must specify at least one goal type"
    assert not (self.velocity_goals and goalsampler_qd_limit is None), "Must specify goalsampler_qd_limit"
    assert not (self.velocity_goals and goalsampler_qd_limit.shape[0] != self.dof), "Must specify correctly configured goalsampler_qd_limit"
    if self.position_goals and self.full_velocity_goals:
         self.goal_x_mask = 1
         self.goal_xd_mask = 1
    elif self.position_goals and self.root_velocity_goals:
        self.goal_x_mask = 1
        self.goal_xd_mask = jp.zeros((self.num_links,3)).at[0].set(1.0)
    elif self.position_goals:
        self.goal_x_mask = 1
        self.goal_xd_mask = 0
    elif self.full_velocity_goals:
        self.goal_x_mask = 0
        self.goal_xd_mask = 1
    else:
        assert self.position_goals, "Cannot only specify root_velocity_goals"

    # Goal sampling attributes
    if env_name == 'ant':
        self.end_effector_idx = [2,4,6,8]
        self.goal_z_cond = jp.array([0.078, 1.3])
        self.goal_polar_cond = jp.pi/12
        self.goal_contact_cond = [0.078, 0.09]

    goalsampler_q_limit = jp.array(self.sys.dof.limit).T.at[0:3].set(goalsampler_root_pos_range).at[3:6].set(goalsampler_root_rot_range)
    self.goalsampler_limit = jp.concatenate([goalsampler_q_limit, goalsampler_qd_limit])

    # Training attributes
    self.max_goal_dist = max_goal_dist
    self.max_actions_per_node = 1 if env_name == 'ant' else 2
    self.obs_mask = obs_mask
    self.non_actuator_nodes = 0
    self.action_mask = action_mask

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
    goal_distance = self._goal_dist(pipeline_state, goal, frame = 'relative')/self.max_goal_dist

    # Get observation
    obs = self._get_obs(pipeline_state, goal)
    
    # Set metrics
    reward, done, zero = jp.zeros(3)
    metrics = {
        'intrinsic_reward': zero,
        'unhealthy_cost': zero,
        'goal_distance': goal_distance
    }

    info = {'goal': goal}

    return State(pipeline_state, obs, reward, done, metrics, info)


  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    goal = state.info['goal']

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
    goal_distance = self._goal_dist(pipeline_state, goal, frame = 'relative')/self.max_goal_dist

    # Compute rewards: R = ||s-g|| - ||s'-g||
    intrinsic_reward = self._get_intrinsic_reward(
      previous_state=prev_pipeline_state, 
      current_state=pipeline_state,
      goal=goal
      )
    unhealthy_cost = self.unhealthy_cost * is_unhealthy

    # Check if goal reached
    goal_reached = jp.where(
        goal_distance < self.goal_distance_epsilon, x=1.0, y=0.0
    )

    # Compute state observation
    obs = self._get_obs(pipeline_state, goal)
    reward = intrinsic_reward - unhealthy_cost

    if (bool(is_unhealthy) or bool(goal_reached))\
        and self._terminate_when_unhealthy:
      done = 1.0 
    else:
      done = 0.0

    state.metrics.update(
        intrinsic_reward=intrinsic_reward,
        unhealthy_cost=unhealthy_cost
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )


  def _get_obs(self, state: base.State, goal: Goal) -> jp.ndarray:
    """
    Processes a state and goal into observation format

    Args:
        state: dynamic State object, changing every step
        goal: Goal object, containing egocentric Transform goals
          and/or world frame velocity goals

    Returns:
        obs: (num_links, 13 + goal_size) array containing goal observations:
    """
    def _get_obs_x(self, state: Union[base.State, Goal]):
        """Returns world root position (masking XY position) and limb position relative to parent"""
        return world_to_relative(state.x, self.sys, mask_root = True)
    
    def _get_obs_gx(self, goal: Goal, state: base.State):
        """Returns world root position and egocentric limb position"""
        state_x = world_to_relative(state.x, self.sys)
        goal_x = goal.x_rel
        return goal_x.vmap(in_axes = (0, 0)).to_local(state_x) * self.goal_x_mask
    
    def _get_obs_xd(self, state: Union[base.State, Goal]):
        """Returns world root velocity and egocentric limb velocity"""
        return world_to_relative(state.x, self.sys)
    
    def _get_obs_gxd(self, goal: Goal, sxd: base.State):
        """Returns world root velocity and egocentric limb velocity"""
        return goal.xd_rel.__sub__(sxd) * self.goal_xd_mask
          
    sx = concatenate_attrs(_get_obs_x(state))
    sxd = concatenate_attrs(_get_obs_xd(state))
    gx = concatenate_attrs(_get_obs_gx(goal, state)) if self.position_goals else jp.empty((self.num_links, 0))
    gxd = concatenate_attrs(_get_obs_gxd(goal, sxd)) if self.velocity_goals else jp.empty((self.num_links, 0))
    
    s_obs = jp.concatenate([sx, sxd], axis = -1)
    g_obs = jp.concatenate([gx, gxd], axis = -1)

    if self.goal_nodes:
        g_obs = pad(g_obs, self.state_obs_width)
        s_obs = jp.concatenate([s_obs, jp.zeros((self.num_links,1))], axis = -1)
        g_obs = jp.concatenate([s_obs, jp.ones((self.num_links,1))], axis = -1)
        obs = jp.concatenate([s_obs, g_obs], axis = 0)
    else:
        obs = jp.concatenate([s_obs, g_obs], axis=-1)

    return obs


  def _get_intrinsic_reward(self, previous_state: State, current_state: State, goal: Goal) -> float:
    def difference_reward():
      prev_goal_distance= self._goal_dist(previous_state, goal, frame = 'world')
      current_goal_distance = self._goal_dist(current_state, goal, frame = 'world')
      return prev_goal_distance - current_goal_distance  
    def absolute_reward():
      current_goal_distance = self._goal_dist(current_state, goal, frame = 'world')
      return -current_goal_distance
    return jax.lax.cond(
       self.distance_reward == 'difference',
       difference_reward,
       absolute_reward)
  
  def _goal_dist(self, state: base.State, goal: Goal, frame: Literal['world', 'relative']):
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
        state_x = world_to_relative(state.x, self.sys)
        state_xd = world_to_relative(state.xd, self.sys)
        goal_x = goal.x_rel
        goal_xd = goal.xd_rel

    rpos = state_x.pos - goal_x.pos
    rrot = dist_quat(state_x.rot, goal_x.rot)
    rx = concatenate_attrs(base.Transform(rpos, rrot) * self.goal_x_mask)
    rxd = concatenate_attrs(state_xd.__sub__(goal_xd) * self.goal_xd_mask)
    s_minus_g = jp.concatenate([rx, rxd])
    return safe_norm(s_minus_g)
  
  
  def _sample_goal(self, rng: jp.ndarray, state: generalized.base.State):
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
    def _create_goal(self, g: jp.ndarray, state: base.State, sys: base.System) -> Goal:
      """
      Creates a goal state 

      Args:
        g: jp.ndarray normalised goal
        state: environment state

      Returns:
        Goal: goal state
      """
      q = q_spherical_to_quaternion(g[:self.dof], state, sys)
      qd = g[-self.dof:]
      x_world, xd_world = forward(self.sys, q, qd)
      x_world, xd_world = __mul__(x_world, self.goal_x_mask), __mul__(xd_world, self.goal_xd_mask)
      x_rel, xd_rel = world_to_relative(x_world, sys), world_to_relative(xd_world, sys)
      x_rel, xd_rel = __mul__(x_rel, self.goal_x_mask), __mul__(xd_rel, self.goal_xd_mask)
      return Goal(x_world, x_rel, xd_world, xd_rel)

    rng, rng1 = jax.random.split(rng)
    foot_contact_idx = random_ordered_subset(rng1, self.end_effector_idx)

    def sample(carry):
      rng, _ = carry
      rng, rng1 = jax.random.split(rng)
      g = jax.random.uniform(
          rng1, 
          shape=self.goal_size, 
          minval=self.goal_limit[:, 0], 
          maxval=self.goal_limit[:, 1]
        )
      goal = self.create_goal(g, state)
      return rng, goal
    
    def reject(carry):
      rng, goal = carry
      #cpos_foot_z = contact(self.sys, goal.x).pos[foot_contact_idx, 2]
      cpos_foot_z = goal.x_world.pos[foot_contact_idx, 2]
      z = goal.x_world.pos[:,2]
      polar = goal.bpg[6]
      cond = jp.any(z < self.goal_z_cond[0]) | \
        jp.any(z > self.goal_z_cond[1]) | \
        (polar > self.goal_polar_cond) | \
        jp.any(cpos_foot_z > self.goal_contact_cond[1]) | \
        jp.any(cpos_foot_z < self.goal_contact_cond[0])
      return cond
    
    init_g = jax.random.uniform(
        rng, 
        shape=self.goal_size, 
        minval=-1, 
        maxval=1
    )

    init_val = rng, _create_goal(init_g, state)
    goal = jax.lax.while_loop(reject, sample, init_val)[1]

    return goal
