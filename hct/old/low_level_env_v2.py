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

from brax import base, generalized
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward

from etils import epath

import jax
from jax import numpy as jp

from typing import Optional, Literal, Tuple

from absl import logging


class LowLevelEnvV2(PipelineEnv):

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
    morphology: Literal['ant', 'humanoid'] = 'ant_test',
    position_goals: bool = True,
    velocity_goals: bool = False, 
    obs_mask: Optional[jp.ndarray] = None,
    distance_reward: Literal['difference', 'absolute'] = 'absolute',
    terminate_when_goal_reached=True,
    goal_distance_epsilon = 0.0005, 
    architecture_name='MLP',
    architecture_configs=DEFAULT_MLP_CONFIGS, # trial larger network
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
    self.q_limits = jax.tree_map(lambda x: jp.zeros((self.sys.q_size(),)).at[7:].set(x[6:]).at[3].set(1.0), self.sys.dof.limit)
    self.qd_limits = self._get_qd_limits()

    # Reward attributes
    self.distance_reward = distance_reward
    
    # Termination attributes
    self._terminate_when_goal_reached = terminate_when_goal_reached
    self.goal_distance_epsilon = goal_distance_epsilon

    # Goal attributes
    self.position_goals = position_goals
    self.velocity_goals = velocity_goals
    self.max_goal_dist = safe_norm(2*jp.ones(8)) * (self.position_goals + self.velocity_goals)

    # Training attributes
    self.obs_mask = obs_mask
    self.non_actuator_nodes = 0 
    self.action_mask = None
    self.num_nodes = 9

    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name=architecture_name, **architecture_configs)
    self.max_actions_per_node = 1 if self.network_architecture.name=='Transformer' else 8

    self.action_repeat = 1
    self.episode_length = 1000
    self.action_shape = (8, 1)

    logging.info('Environment initialised.')

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
    
    q = jax.random.uniform(
      rng1, (self.sys.q_size(),), minval=self.q_limits[0], maxval=self.q_limits[1]
    )

    qd = 0.5 * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(q, qd)

    goal_q = jax.random.uniform(
      rng3, (self.sys.q_size(),), minval=self.q_limits[0], maxval=self.q_limits[1]
    )
    goal_qd = jax.random.uniform(
      rng4, (self.sys.qd_size(),), minval=self.qd_limits[0], maxval=self.qd_limits[1]
    )

    # Sample and set goal
    goal_x, goal_xd = forward(self.sys, goal_q, goal_qd)
    gja, gjv = self.get_ja_jv(goal_q, goal_qd)

    goal = Goal(goal_q, goal_qd, goal_x, goal_xd, ja=gja, jv=gjv)

    # Get observation
    obs, goal_dist = self.get_obs(pipeline_state, goal)
    
    # Set metrics
    reward, done = jp.zeros(2)

    if self.distance_reward == 'absolute':
      reward= -goal_dist 
      
    metrics = {
      'reward': reward,
      'goal_distance': goal_dist,
      'goal_distance_normalised': goal_dist/self.max_goal_dist,
      'goal_reached': 0.0
    }

    info = {
      'goal': goal
    }

    return State(pipeline_state, obs, reward, done, metrics, info)


  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    if action.shape[-1] == 1:
      action = jp.squeeze(action, axis=-1) 

    goal = state.info['goal']
    prev_goal_dist = state.metrics['goal_distance']

    # Take action
    prev_pipeline_state = state.pipeline_state
    pipeline_state = self.pipeline_step(prev_pipeline_state, action)

    # Compute state observation
    obs, goal_dist = self.get_obs(pipeline_state, goal)
    goal_dist_normalised = goal_dist/self.max_goal_dist

    # Check if goal reached
    goal_reached = jp.where(
      goal_dist_normalised <= self.goal_distance_epsilon, x=1.0, y=0.0
    )

    # Compute rewards: 
    if self.distance_reward == 'absolute':
      reward = -goal_dist
    else:
      reward = (prev_goal_dist- goal_dist)/self.dt 

    done = 0.0 + goal_reached
    
    state.metrics.update(
      reward=reward,
      goal_distance=goal_dist,
      goal_distance_normalised=goal_dist_normalised,
      goal_reached=1.0*goal_reached
    )

    return state.replace(
      pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
    
  def get_obs(self, state: base.State, goal: Goal) -> jp.ndarray:
    """
    Processes a state and goal into observation format

    Args:
        state: dynamic State object, changing every step
        goal: Goal object, containing egocentric Transform goals
          and/or world frame velocity goals

    Returns:
        obs: (num_links, 13 + goal_size) array containing goal observations:
    """
    ja, jv = self.get_ja_jv(state.q, state.qd)
    gja = (goal.ja - ja) * self.position_goals
    gjv = (goal.jv - jv) * self.velocity_goals

    obs = jp.concatenate([jp.expand_dims(x, axis=1) for x in [ja, jv, gja, gjv]], axis=-1)
  
    if self.network_architecture.name == 'MLP':
      if not self.position_goals:
        obs = obs[:, jp.array([0, 1, 3])]
      elif not self.velocity_goals:
        obs = obs[:, jp.array([0, 1, 2])]
      obs = obs.reshape(*obs.shape[:-2], -1)

    goal_dist = self.goal_dist(ja, jv, goal)
    return obs, goal_dist
  
  def get_ja_jv(self, q, qd):
    normalize_to_range_vmap = jax.vmap(normalize_to_range, in_axes=(0, 0, 0, None, None))
    ja = normalize_to_range_vmap(q, self.q_limits[0], self.q_limits[1], -1, 1)[6:]
    jv = normalize_to_range_vmap(qd, self.qd_limits[0], self.qd_limits[1], -1, 1)[5:]
    return ja, jv
  
  def goal_dist(self, ja, jv, goal):
    dist_ja = (ja - goal.ja) * self.position_goals
    dist_jv = (jv - goal.jv) * self.velocity_goals
    return safe_norm(dist_ja + dist_jv)
    
  def _get_qd_limits(self):

    filepath = '/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/envs/ranges/'

    variant = 'Low level v2'
    
    filename = f'{filepath}/{variant}'
    
    if os.path.isfile(filename):
      return model.load(filename)
    
    test_rollout, _ = AntTest().test_rollout()
    rollout_qd = jp.stack([state.qd for state in test_rollout])

    qd_limit = (
      jp.min(rollout_qd, axis=0).at[0:6].set(0), 
      jp.max(rollout_qd, axis=0).at[0:6].set(0)
    )

    model.save(filename, qd_limit)
    return qd_limit
