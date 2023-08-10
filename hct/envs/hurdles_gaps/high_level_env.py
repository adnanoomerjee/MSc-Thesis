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
from hct.envs.high_level_env import HighLevelEnv
from hct.envs.hurdles_gaps.reward_function import reward

from brax import base, generalized
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward

from etils import epath

import jax
from jax import numpy as jp

from typing import Optional, Literal, Tuple

from absl import logging


class HighLevelHurdlesGapsEnv(HighLevelEnv):

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
    task: Literal['ant_hurdles', 'ant_gaps'],
    mid_level_modelpath: str, 
    reward_milestone = 100,
    reward_type: Literal['sparse', 'dense'] = 'sparse',
    reward_movement: Literal['position', 'velocity'] = 'velocity',
    action_repeat=1,
    mid_level_goal_root_pos_range: jp.ndarray = jp.array([[-1,1], [-1,1], [-0.25, 0.45]]),
    architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
    healthy_z_range=(0.0, 10.0),
    **kwargs
  ):
    

    super().__init__(
      task=task,
      mid_level_modelpath=mid_level_modelpath,
      action_repeat=action_repeat,
      mid_level_goal_root_pos_range=mid_level_goal_root_pos_range,
      architecture_configs=architecture_configs,
      **kwargs)

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')

    # Reward attributes
    self.reward_milestone = reward_milestone
    self.reward_movement = reward_movement
    self.reward_type = reward_type 

    # Termination attributes
    self._terminate_when_unhealthy = True if self.reward_type == 'dense' and self.reward_movement == 'velocity' else False
    self._healthy_z_range = healthy_z_range

    self.episode_length = 1500/self.mid_level_env.action_repeat

    obstacle_positions = [4.5, 9.25, 17, 21.4, 27.65, 34.95, 41.05, 47.35, 52, 57.8, 62]
    obstacle_sizes = [0.5, 0.25, 0.5, 0.9, 0.35, 0.95, 0.95, 0.35, 0.7, 0.5, 0]

    self.endpoint = 62.0

    self.obstacle_completion_coords = jp.array(
      [y_position + y_size for y_position, y_size in zip(obstacle_positions, obstacle_sizes)]
    )


  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    q = self.sys.init_q 
    qd = jp.zeros((self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(q, qd)

    # Get observation
    obs = self.get_obs(pipeline_state)
    
    # Set metrics
    done, reward = jp.array([0.0, -1.0])

    metrics = {
      'reward': reward,
      'obstacles_complete': 0.0,
      'distance_to_end': 62 - pipeline_state.x.pos[0, 1],
      'is_unhealthy': 0.0,
      'task_complete': 0.0
    }

    info = {
      'rng': rng,
      'goal': None
    }

    return State(pipeline_state, obs, reward, done, metrics, info)


  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    
    if action.shape[-1] == 1:
      action = jp.squeeze(action, axis=-1) 

    obstacles_complete = state.metrics['obstacles_complete']

    rng = state.info['rng']

    prev_pipeline_state = state.pipeline_state

    # Take action
    pipeline_state, rng = self.take_action(prev_pipeline_state, action, rng)

    # Check if unhealthy
    is_unhealthy = jp.where(rotate(jp.array([0, 0, 1]), pipeline_state.x.rot[0])[-1] < 0, x=1.0, y=0.0)
    is_unhealthy = jp.where(pipeline_state.x.pos[0][-1] < 0, x=1.0, y=is_unhealthy)

    reward, obstacles_complete = self._reward(obstacles_complete, pipeline_state)

    # Get observation
    obs = self.get_obs(pipeline_state)

    # Check if goal reached
    goal_reached = pipeline_state.x.pos[0, 1] >= 62

    obstacles_complete = obstacles_complete * (1 - goal_reached) + goal_reached * 10
    distance_to_end = (62 - pipeline_state.x.pos[0, 1]) * (1 - goal_reached) + goal_reached * 0.0

    if self._terminate_when_unhealthy:
      done = 0.0 + jp.logical_or(is_unhealthy, goal_reached)
    else:
      done = 0.0 + goal_reached
    
    state.metrics.update(
      reward=reward,
      obstacles_complete=obstacles_complete,
      distance_to_end=distance_to_end,
      is_unhealthy=1.0*is_unhealthy,
      task_complete=1.0*goal_reached
    )

    state.info.update(
      rng=rng
    )
    
    return state.replace(
      pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _reward(self, obstacles_complete, state: base.State):
    return reward(self, obstacles_complete, state)

  


      


'''
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
    randint = jax.random.randint(rng1,shape=(1,), minval = 0, maxval = 4)
    randchoice = jax.random.choice(rng2, jp.array([self.goal_contact_cond, self.goal_z_cond[1]]), shape=(1,), p = jp.array([0.9, 0.1]))
    goal_contact_cond = jp.full((4,), self.goal_z_cond[1]).at[randint].set(randchoice)

    def _create_goal(g: jp.ndarray, state: base.State) -> Goal:
      """
      Creates a goal state 

      Args:
        g: jp.ndarray normalised goal
        state: environment state

      Returns:
        Goal: goal state
      """
      q = q_spherical_to_quaternion(g[:self.dof], state, self.sys)
      qd = g[-self.dof:]
      x_world, xd_world = forward(self.sys, q, qd)
      x_world, xd_world = x_world * self.goal_x_mask, xd_world * self.goal_xd_mask
      x_rel, xd_rel = world_to_relative(x_world, self.sys), world_to_relative(xd_world, self.sys)
      x_rel, xd_rel = x_rel * self.goal_x_mask, xd_rel * self.goal_xd_mask
      return Goal(g, q, qd, x_world, x_rel, xd_world, xd_rel)

    def _sample(carry):
      rng, _ = carry
      rng, rng1 = jax.random.split(rng)
      g = jax.random.uniform(
        rng1, 
        shape=self.goal_size, 
        minval=self.goalsampler_limit[:, 0], 
        maxval=self.goalsampler_limit[:, 1]
      )
      goal = _create_goal(g, state)
      return rng, goal
    
    def _reject(carry):
      rng, goal = carry
      #cpos_foot_z = contact(self.sys, goal.x).pos[foot_contact_idx, 2]
      foot_z = goal.x_world.pos[self.end_effector_idx, 2]
      z = goal.x_world.pos[:,2]
      polar = goal.g[4]
      cond = \
        jp.any(z < self.goal_z_cond[0]) | jp.any(z > self.goal_z_cond[1]) | \
        (polar > self.goal_polar_cond) | \
        jp.any(foot_z > goal_contact_cond)
      return cond
    
    init_g = jax.random.uniform(
      rng, 
      shape=self.goal_size, 
      minval=self.goalsampler_limit[:, 0], 
      maxval=self.goalsampler_limit[:, 1]
    )

    init_val = rng, _create_goal(init_g, state)
    goal = jax.lax.while_loop(_reject, _sample, init_val)[1]

    return goal

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

    def _create_goal(g: jp.ndarray, state: base.State) -> Goal:
      """
      Creates a goal state 

      Args:
        g: jp.ndarray normalised goal
        state: environment state

      Returns:
        Goal: goal state
      """
      
      q = q_spherical_to_quaternion(g[:self.dof], state, self.sys)
      qd = g[-self.dof:]
      x_world, xd_world = forward(self.sys, q, qd)
      x_world, xd_world = x_world * self.goal_x_mask, xd_world * self.goal_xd_mask
      x_rel, xd_rel = world_to_relative(x_world, self.sys), world_to_relative(xd_world, self.sys)
      x_rel, xd_rel = x_rel * self.goal_x_mask, xd_rel * self.goal_xd_mask
      return Goal(g, q, qd, x_world, x_rel, xd_world, xd_rel)

    rng, rng1, rng2 = jax.random.split(rng, 3)
    g = jax.random.uniform(
      rng, 
      shape=self.goal_size, 
      minval=self.goalsampler_limit[:, 0], 
      maxval=self.goalsampler_limit[:, 1]
    )
    goal = _create_goal(g, state)
    z = goal.x_world.pos[:,2]
    z_root = z[0] - jp.min(z) + self.goal_z_cond[0] 
    z_root = z_root + jax.random.choice(rng1, jp.array([0, 1]), p=jp.array([0.9, 0.1])) * jax.random.uniform(rng2, minval=z_root, maxval=self.goal_z_cond[1]-z_root)

    q = goal.q.at[2].set(z_root)
    qd = goal.qd
    x_world, xd_world = forward(self.sys, q, qd)
    x_world, xd_world = x_world * self.goal_x_mask, xd_world * self.goal_xd_mask
    x_rel, xd_rel = world_to_relative(x_world, self.sys), world_to_relative(xd_world, self.sys)
    x_rel, xd_rel = x_rel * self.goal_x_mask, xd_rel * self.goal_xd_mask   

    return Goal(g, q, qd, x_world, x_rel, xd_world, xd_rel)

    
'''