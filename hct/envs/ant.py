# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Trains an ant to run in the +x direction."""
#import sys
#sys.path.insert(0, "/nfs/nhome/live/aoomerjee/MSc-Thesis/")

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.kinematics import forward
from brax.io import mjcf, html
from etils import epath
import jax
from jax import numpy as jp

from hct.envs.tools import world_to_relative, safe_norm, concatenate_attrs, dist_quat, quaternion_to_spherical

from typing import Literal


class Ant(PipelineEnv):



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
      ctrl_cost_weight=0.5,
      use_contact_forces=False,
      contact_cost_weight=5e-4,
      healthy_reward=1.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.2, 1.0),
      contact_force_range=(-1.0, 1.0),
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs,
  ):
    path = epath.resource_path('hct') / f'envs/assets/ant_original.xml'
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

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._contact_force_range = contact_force_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

    if self._use_contact_forces:
      raise NotImplementedError('use_contact_forces not implemented.')

  def reset(self, rng: jp.ndarray, limit_id: int) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)
    
    joint_angles = jax.lax.select(limit_id==0, jp.array(self.sys.dof.limit[0]), jp.array(self.sys.dof.limit[1]))
    q = jp.concatenate([jp.array([0,0,0,1,0,0,0]), joint_angles[6:]])
    qd = jp.zeros((self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)

    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_survive': zero,
        'reward_ctrl': zero,
        'reward_contact': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'forward_reward': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
    forward_reward = velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
    is_healthy = jp.where(
        pipeline_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy
    )
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    contact_cost = 0.0

    obs = self._get_obs(pipeline_state)
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        reward_forward=forward_reward,
        reward_survive=healthy_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        forward_reward=forward_reward,
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
    """Observe ant body position and velocities."""
    qpos = pipeline_state.q
    qvel = pipeline_state.qd

    if self._exclude_current_positions_from_observation:
      qpos = pipeline_state.q[2:]

    return jp.concatenate([qpos] + [qvel])
  
  def _limb_dist(
      self, 
      state1: base.State, 
      state2: base.State, 
      limb_id: int,  
      frame: Literal['world', 'relative']):
    """
    Computes distance d(s,g) between state and goal in world frame, 
    accounting for quaternion double cover.

    dist(s,g) = ||s-g||
    """
    if frame == 'world':
        state1_x = state1.x.take(limb_id)
        state1_xd = state1.xd.take(limb_id) 
        state2_x = state2.x.take(limb_id)
        state2_xd = state2.xd.take(limb_id)
    else:
        state1_x = world_to_relative(state1.x.take(limb_id), self.sys)
        state1_xd = world_to_relative(state1.xd.take(limb_id), self.sys)
        state2_x = world_to_relative(state2.x.take(limb_id), self.sys)
        state2_xd = world_to_relative(state2.xd.take(limb_id), self.sys)

    rpos = state1_x.pos - state2_x.pos
    rrot = dist_quat(state1_x.rot, state2_x.rot)
    rx = concatenate_attrs(base.Transform(rpos, rrot))
    rxd = concatenate_attrs(state1_xd.__sub__(state2_xd))
    x_dist = safe_norm(rx) 
    xd_dist = safe_norm(rxd)
    return x_dist, xd_dist
  

  def move_limb(self, limb_id, actuator_force):
    return jp.zeros(((self.action_size,),)).at[limb_id].set(actuator_force)


  def get_limb_x_dist(self, x0: base.Transform, x1: base.Transform, limb_id):
    x0 = x0.take(limb_id)
    x1 = x1.take(limb_id)
    rpos = x0.pos - x1.pos
    rrot = dist_quat(x0.rot, x1.rot)
    rx = concatenate_attrs(base.Transform(rpos, rrot))
    x_dist = safe_norm(rx) 
    return x_dist

  def get_limb_xd_dist(self, xd0: base.Transform, xd1: base.Transform, limb_id):
    xd0 = xd0.take(limb_id)
    xd1 = xd1.take(limb_id)
    rxd = concatenate_attrs(xd0.__sub__(xd1))   
    return safe_norm(rxd)

  def get_limb_ranges(self):
    
    quaternion_to_spherical_vmap = jax.vmap(quaternion_to_spherical, in_axes=0)
    jit_env_reset = jax.jit(self.reset)
    jit_env_step = jax.jit(self.step)
    jit_move_limb = jax.jit(self.move_limb)

    q0 = jp.concatenate([jp.array([0,0,0,1,0,0,0]), jp.array(self.sys.dof.limit[0])[6:]])
    q1 = jp.concatenate([jp.array([0,0,0,1,0,0,0]), jp.array(self.sys.dof.limit[1])[6:]])

    qd0 = jp.zeros((self.sys.qd_size(),))   

    x0 = world_to_relative(forward(self.sys, q0, qd0)[0], self.sys)
    x1= world_to_relative(forward(self.sys, q1, qd0)[0], self.sys)

    xd0 = base.Motion.zero(shape = (self.sys.num_links(),))
    xd0 = world_to_relative(xd0, self.sys)

    upper_leg_dists = {}
    lower_leg_dists = {}

    upper_leg_dists['max_x_dist'] = self.get_limb_x_dist(x0, x1, 1)
    lower_leg_dists['max_x_dist'] = self.get_limb_x_dist(x0, x1, 2)

    rollout = []
    rollout_rel = []
    rng = jax.random.PRNGKey(seed=1)
    
    for ranges in (upper_leg_dists, lower_leg_dists):

      if ranges == upper_leg_dists:
        limb_id = 1
      else:
        limb_id = 2

      ranges['max_xd_dist'] = 0

      for limit_id in (0,1):

        for actuator_force in (-1,1,-1,1):

          rng, rng1 = jax.random.split(rng)

          state = jit_env_reset(rng=rng, limit_id=limit_id)
          print(state)

          for _ in range(60):

            rollout.append(state.pipeline_state)

            pipeline_state_rel = state.pipeline_state
            pipeline_state_rel.x = world_to_relative(state.pipeline_state.x, self.sys)
            pipeline_state_rel.xd = world_to_relative(state.pipeline_state.xd, self.sys)
            rollout_rel.append(state.pipeline_state_rel)

            xd_dist = self.get_limb_xd_dist(xd0, pipeline_state_rel.xd , limb_id)
            if xd_dist > ranges['max_xd_dist']:
              ranges['max_xd_dist'] = xd_dist

            act = jit_move_limb(limb_id=limb_id, actuator_force=actuator_force)
            state = jit_env_step(state, act)

    rollout_rel_pos = jp.stack([state.x.pos for state in rollout_rel])
    rollout_rel_rot = jp.stack([quaternion_to_spherical_vmap(state.x.rot) for state in rollout_rel])
    rollout_rel_vel = jp.stack([state.xd.vel for state in rollout_rel])
    rollout_rel_ang = jp.stack([state.xd.ang for state in rollout_rel])

    pos_ranges = jp.min(rollout_rel_pos, axis=0), jp.max(rollout_rel_pos, axis=0)
    rot_ranges = jp.min(rollout_rel_rot, axis=0), jp.max(rollout_rel_rot, axis=0)
    vel_ranges = jp.min(rollout_rel_vel, axis=0), jp.max(rollout_rel_vel, axis=0)
    ang_ranges = jp.min(rollout_rel_ang, axis=0), jp.max(rollout_rel_ang, axis=0)

    return_dict = {
      'upper_leg_dists': upper_leg_dists, 
      'lower_leg_dists': lower_leg_dists,
      'pos_ranges': pos_ranges, 
      'rot_ranges': rot_ranges, 
      'vel_ranges': vel_ranges, 
      'ang_ranges': ang_ranges, 
      'rollout': rollout
    }
    return upper_leg_dists, lower_leg_dists, pos_ranges, rot_ranges, vel_ranges, ang_ranges, rollout




