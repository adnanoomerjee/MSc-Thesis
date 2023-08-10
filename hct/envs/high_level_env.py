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



class HighLevelEnv(PipelineEnv):

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
      task,
      mid_level_modelpath: str,
      action_repeat=5,
      mid_level_goal_root_pos_range: jp.ndarray = jp.array([[-1,1], [-1,1], [-0.25, 0.45]]),
      architecture_configs = DEFAULT_MLP_CONFIGS, # trial larger network
      **kwargs
  ):
    
    mid_level_network = load(f"{mid_level_modelpath}/network")
    mid_level_params = load(f"{mid_level_modelpath}/model_params")
    mid_level_make_inference_fn = load(f"{mid_level_modelpath}/make_inference_fn")
    mid_level_env = load(f"{mid_level_modelpath}/env")

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    self.parameters = {arg: values[arg] for arg in args}
    self.parameters.pop('self')

    logging.info('Initialising environment...')

    path = epath.resource_path('hct') / f'envs/assets/{task}.xml'
    sys = mjcf.load(path)

    mid_level_env.sys = sys

    n_frames = 5

    backend = mid_level_env.backend

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
    
    self.mid_level_env = mid_level_env
    self.mid_level_policy = mid_level_make_inference_fn(mid_level_network)(
        mid_level_params,
        train=False,
        obs_mask=mid_level_env.obs_mask,
        action_mask=mid_level_env.action_mask,
        non_actuator_nodes=mid_level_env.non_actuator_nodes,
        deterministic=False
    )
    self.low_level_env = mid_level_env.low_level_env
    self.low_level_policy = mid_level_env.low_level_policy

    # Agent attributes
    self.dof = mid_level_env.dof
    self.num_links = mid_level_env.num_links
    self.link_parents = (-1)

    # Training attributes
    self.obs_mask = None
    self.non_actuator_nodes = None
    self.num_nodes = 1

    # Goal attributes
    self.position_goals = mid_level_env.position_goals
    self.velocity_goals = mid_level_env.velocity_goals
    self.root_velocity_goals = mid_level_env.root_velocity_goals
    self.full_velocity_goals = mid_level_env.full_velocity_goals

    # Observation attributes
    self.state_obs_width = mid_level_env.state_obs_width

    if self.position_goals and self.full_velocity_goals:
      self.max_actions_per_node = 60
    elif self.position_goals and self.root_velocity_goals:
      self.max_actions_per_node = 36
    else:
      self.max_actions_per_node = 30

    self.action_mask = None
    
    mid_level_pos_range_0 = self.mid_level_env.pos_range[0].at[0].set(mid_level_goal_root_pos_range[:,0])
    mid_level_pos_range_1 = self.mid_level_env.pos_range[1].at[0].set(mid_level_goal_root_pos_range[:,1])

    self.mid_level_goal_ranges = jp.concatenate(
        [
          jp.array((mid_level_pos_range_0, mid_level_pos_range_1)),
          jp.squeeze(jp.array(self.mid_level_env.rot_range)),
          jp.array(self.mid_level_env.vel_range_range),
          jp.array(self.mid_level_env.ang_range)
        ],
        axis=-1
      )

    # Network architecture
    self.network_architecture = NetworkArchitecture.create(name='MLP', **architecture_configs)
    self.action_repeat = action_repeat

    logging.info('Environment initialised.')

  def take_action(self, cur_pipeline_state: base.State, mid_level_goal: jp.array, rng):

    rng, rng1 = jax.random.split(rng)

    mid_level_goal = self.get_mid_level_goal(mid_level_goal)
    mid_level_obs = self.mid_level_env.get_obs(self.mid_level_env._average_limbs(cur_pipeline_state), mid_level_goal) 
    low_level_goal, _ = self.mid_level_policy(mid_level_obs, rng1) 
    low_level_goal = self.mid_level_env._get_low_level_goal(low_level_goal) 

    def low_level_goal_repeat(carry, unused_nstate):
      pipeline_state, rng = carry
      rng, rng1 = jax.random.split(rng)
      low_level_obs = self.low_level_env.get_obs(pipeline_state, low_level_goal)
      action, _ = self.low_level_policy(low_level_obs, rng1) 
      nstate = self.pipeline_step(pipeline_state, action) 
      return (nstate, rng), nstate
    
    carry_init = cur_pipeline_state, rng
    (pipeline_state, rng), _ = jax.lax.scan(
      low_level_goal_repeat, carry_init, (), length=self.mid_level_env.action_repeat)
    
    return pipeline_state, rng
  

  def get_mid_level_goal(self, g):

    if (self.position_goals and self.velocity_goals) or self.root_velocity_goals:
      root_goals = jp.expand_dims(g[:12], axis=0)
      limb_goals = g[12:].reshape(4, g[6:].shape[-1]//4)
      goals = jp.concatenate([root_goals, limb_goals], axis=-2)
    elif self.root_velocity_goals:
      root_goals = jp.expand_dims(g[:12], axis=0)
      limb_goals = g[12:].reshape(4, g[6:].shape[-1]//4)
      limb_goals = pad(limb_goals, root_goals.shape[-1])
      goals = jp.concatenate([root_goals, limb_goals], axis=-2)
    else:
      root_goals = jp.expand_dims(g[:6], axis=0)
      limb_goals = g[6:].reshape(4, g[6:].shape[-1]//4)
      goals = jp.concatenate([root_goals, limb_goals], axis=-2)

    if self.position_goals and not self.velocity_goals:
      goals = pad(goals, 12)
    elif self.velocity_goals_goals and not self.position_goals:
      goals = jp.concatenate([jp.zeros((5,6)) + goals], axis=-1)

    goals_unnormalised = unnormalize_to_range(
      goals,
      self.mid_level_goal_ranges[0],
      self.mid_level_goal_ranges[1],
      -1,
      1
    )

    pos_goals = goals_unnormalised[:, 0:3]
    rot_goals = goals_unnormalised[:, 3:6]
    vel_goals = goals_unnormalised[:, 6:9]
    ang_goals = goals_unnormalised[:, 9:12]

    x_rel = base.Transform(pos_goals, rot_goals)
    xd_rel = base.Motion(vel_goals, ang_goals)

    return Goal(None, None, None, None, x_rel, xd_rel, None)


  def get_obs(self, state: base.State) -> jp.ndarray:
    """Return observation input tensor"""

    spos = jp.mean(state.x.pos, axis=0)
    srot = state.x.rot[0]
    svel = jp.mean(state.xd.vel, axis=0)
    sang = state.xd.ang[0]

    return jp.concatenate([spos, srot, svel, sang], axis = -1)
  

 