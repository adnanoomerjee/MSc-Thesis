# pylint:disable=g-multiple-import
"""Creates an environment for the lowest level of a hierarchical framework"""
import sys
sys.path.insert(0, "/nfs/nhome/live/aoomerjee/MSc-Thesis/")

from hct.envs.observer import Observer
from hct.envs.goal import GoalConstructor, Goal

from brax import base, math
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import world_to_joint

from etils import epath

import jax
from jax import numpy as jp

import multipledispatch



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


  def __init__(
      self,
      configs: dict = {},
      env_name: str = 'ant',
      unhealthy_cost=1.0,
      terminate_when_unhealthy=True,
      terminate_when_goal_reached=True,
      healthy_z_range=(0.2, 1.0),
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
      # TODO: does the same actuator strength work as in spring
      sys = sys.replace(
          actuator=sys.actuator.replace(
              gear=200 * jp.ones_like(sys.actuator.gear)
          )
      )

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)
    self.observer = Observer(sys=sys, configs = configs)
    self.goalconstructor = GoalConstructor(sys=sys, configs = configs)
    self._unhealthy_cost = unhealthy_cost
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._terminate_when_goal_reached = terminate_when_goal_reached
    self._goal_distance_epsilon = goal_distance_epsilon
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale


  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    low, hi = self.observer.goal_q_limit[:,0], self.observer.goal_q_limit[:,1]

    q = self.sys.init_q 
    qd = 0 * hi * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(q, qd)

    goal = self.goalconstructor.sample_goal(rng2, pipeline_state)
    pipeline_state = self.pipeline_init(goal.q, qd)

    # Sample and set goal
    #goal = self.goalconstructor.sample_goal(rng3, pipeline_state)

    # Get observation
    obs = self._get_obs(pipeline_state, goal)
    
    # Set metrics
    reward, done, zero = jp.zeros(3)
    metrics = {
        'intrinsic_reward': zero,
        'unhealthy_cost': zero,
    }
    info = {'goal': goal}

    return State(pipeline_state, obs, reward, done, metrics, info)


  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    goal = state.info['goal']

    # Take action
    pipeline_state_1 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state_1, action)

    # Check if unhealthy
    min_z, max_z = self._healthy_z_range
    is_unhealthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, x=1.0, y=0.0)
    is_unhealthy = jp.where(
        pipeline_state.x.pos[0, 2] > max_z, x=1.0, y=is_unhealthy
    )

    # Compute goal distances
    goal_distance_1 = self._goal_distance(pipeline_state_1)
    goal_distance = self._goal_distance(pipeline_state)

    # Compute rewards: R = ||s-g|| - ||s'-g||
    intrinsic_reward = goal_distance_1 - goal_distance
    unhealthy_cost = self._unhealthy_cost * is_unhealthy

    # Check if goal reached
    goal_reached = jp.where(
        goal_distance < self._goal_distance_epsilon, x=1.0, y=0.0
    )

    # Compute state
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

  def _get_obs(self, pipeline_state: base.State, goal: Goal) -> jp.ndarray:
    """Return observation input tensor"""
    return self.observer.get_obs(pipeline_state, goal)

  def _goal_distance(self, obs: jp.ndarray):
    if self.observer.goal_nodes:
      g_obs = obs[-self.observer.num_links:]
    else:
      obs[:,self.obs_width:]

    if self.observer.q_goals:
      s_minus_g = self.observer\
          .vmap(in_axes = (None, 0, 0))\
            .dist_q_goals(g_obs, self.observer.optimal_goal_obs)
      dist = math.safe_norm(s_minus_g, axis=(0,1))
    else:
      dist = math.safe_norm(g_obs - self.observer.optimal_goal_obs, axis=(0,1))
    return dist

