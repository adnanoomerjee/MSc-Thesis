"""
Defines Observer and Observation classes
"""
from hct.envs.goal import GoalConstructor, Goal
from hct.envs.math import world_to_egocentric, dist_quat
from hct.envs.env_tools import EnvTools, zero_if_non_hinge, concatenate_attrs, pad
from brax.math import safe_norm

from brax import base
from brax import envs

import jax
from jax import numpy as jp

from flax import struct

from typing import Any, Dict, List, Tuple, Union, Optional, Literal


class Observer(EnvTools):
   """
   API for observing a brax system
   
   TODO:

   proj function, for projecting a state into mapped goal space g

   rework functions to accomomdate goal space g attribute in goal

   """
   def __init__(
         self,
         sys: base.System,
         configs: dict,
   ):
      super().__init__(**configs, sys=sys)
   
   def _get_obs_x(self, state: Union[base.State, Goal]):
      """Returns world root position (masking XY position) and egocentric limb position"""
      root = state.x.take(0)
      mask = base.Transform(jp.array([0.0, 0.0, 1.0]), jp.array([1.0, 1.0, 1.0, 1.0]))
      root = self.__mul__(root, mask)
      return world_to_egocentric(state.x).index_set(0, root)
   
   def _get_obs_gx(self, goal: Goal, state: base.State):
      """Returns world root position and egocentric limb position"""
      _sroot = state.x.take(0)
      _sx = world_to_egocentric(state.x).index_set(0, _sroot)
      groot = goal.x.take(0)
      gx = world_to_egocentric(goal.x).index_set(0, groot)
      return gx.vmap(in_axes = (0, 0)).to_local(_sx)
   
   def _get_obs_xd(self, state: Union[base.State, Goal]):
      """Returns world root velocity and egocentric limb velocity"""
      root_xd = state.xd.take(0)
      return state.xd.__sub__(root_xd)
   
   def _get_obs_gxd(self, goal: Goal, sxd: base.State):
      """Returns world root velocity and egocentric limb velocity"""
      return self._get_obs_xd(goal).__sub__(sxd) * goal.xd_mask

   def get_joint_angles(self, state: Union[base.State, Goal]):
      '''Returns normalised joint angles of each joint'''
      ja = self.scan_link_types(zero_if_non_hinge, 'q', 'l', state.q)
      ja = self\
         .vmap(in_axes = (None, 0, 0, 0, None, None))\
            .normalize_to_range(ja, self.jr[:,0], self.jr[:,1], -1, 1)
      return jp.expand_dims(ja, -1)

   def get_joint_velocities(self, state: Union[base.State, Goal]):
      """Returns joint velocity of each joint"""
      return jp.expand_dims(
         self.scan_link_types(zero_if_non_hinge, 'd', 'l', state.qd), -1
      )
   
   def get_obs(self, state: base.State, goal: Goal) -> jp.ndarray:
      """
      Processes a state and goal into observation format

      Args:
         state: dynamic State object, changing every step
         goal: Goal object, containing egocentric Transform goals
            and/or world frame velocity goals

      Returns:
         obs: (num_links, 15 + goal_size) array containing goal observations:
      """      
      sx = concatenate_attrs(self._get_obs_x(state))
      sxd = concatenate_attrs(self._get_obs_xd(state))
      sja = self.get_joint_angles(state) if self.joint_obs else None
      sjv = self.get_joint_velocities(state) if self.joint_obs else None
      gx = concatenate_attrs(self._get_obs_gx(goal, state)) if self.q_goals else jp.empty((self.num_links, 0))
      gxd = concatenate_attrs(self._get_obs_gxd(goal, sxd)) if self.qd_goals else jp.empty((self.num_links, 0))
      gja = self.get_joint_angles(goal) - sja if self.q_goals and self.joint_obs else jp.empty((self.num_links, 0))
      gjv = self.get_joint_velocities(goal) - sjv if self.full_qd_goals and self.joint_obs else jp.empty((self.num_links, 0))
      s_obs = jp.concatenate([sx, sxd, sja, sjv], axis = -1)
      g_obs = jp.concatenate([gx, gxd, gja, gjv], axis = -1)
      if self.goal_nodes:
         g_obs = pad(g_obs, self.obs_width)
         s_obs = jp.concatenate([s_obs, jp.zeros((self.num_links,1))], axis = -1)
         g_obs = jp.concatenate([s_obs, jp.ones((self.num_links,1))], axis = -1)
         obs = jp.concatenate([s_obs, g_obs], axis = 0)
      else:
         obs = jp.concatenate([s_obs, g_obs], axis=-1)
      return obs

   def dist(self, state: base.State, goal: Goal):
      """
      Computes distance d(s,g) between state and goal in world frame, 
      accounting for quaternion double cover.

      dist(s,g) = ||s-g||
      """
      rpos = state.x.pos - goal.x.pos
      rrot = dist_quat(state.x.rot, goal.x.rot)
      rx = concatenate_attrs(base.Transform(rpos, rrot) * self.goal_x_mask)
      rxd = concatenate_attrs(state.xd.__sub__(goal.xd) * self.goal_xd_mask)
      s_minus_g = jp.concatenate([rx, rxd])
      return safe_norm(s_minus_g)

   

   
   """

@struct.dataclass
class Observation(EnvTools):

   """'''Dynamic observation that changes every pipeline step.

   Attributes:
      pos: (num_links, 3) link position in egocentric frame, with z coordinate of root in world frame
      rot: (num_links, 4) link quaternion rotation in world frame
      vel: (num_links, 3) link velocity in world frame
      ang: (num_links, 3) link angular velocity in world frame
      ja: (num_links,) normalised joint angles
      jv: (num_links,) joint velocities
      jr: (num_links, 2) normalised joint ranges
      g_pos: (num_links, 3) goal position in link frame
      g_rot: (num_links, 4) goal quaternion rotation in link frame
      g_vel: (num_links, 3) goal velocity in egocentric frame
      g_ang: (num_links, 3) goal angular velocity in egocentric frame
      g_ja: (num_links,) goal normalised joint angles
      g_jv: (num_links,) goal joint velocities
      g_jr: (num_links, 2) goal normalised joint ranges
      d_goal: (num_links, 3 or 6) distance to goal position (and velocity if specified)'''
   
   """
   pos: jp.ndarray
   rot: jp.ndarray
   vel: jp.ndarray
   ang: jp.ndarray
   ja: jp.ndarray
   jv: jp.ndarray
   jr: jp.ndarray
   g_pos: Optional[jp.ndarray]
   g_rot: Optional[jp.ndarray]
   g_vel: Optional[jp.ndarray]
   g_ang: Optional[jp.ndarray]
   g_ja: Optional[jp.ndarray]
   g_jv: Optional[jp.ndarray]
   
   def format_for_input(self):
      """'''
      Concatenates the observation attributes into a tensor for input
      into transformer model

      Returns:i
         2D tensor, each row corresponding to agent node with features.''' """
      return jp.concatenate([v for k, v in vars(self).items()], axis=-1)
   def _get_egocentric_position(self, state: base.State):
      pos = state.x.pos
      z_root = pos[0, 2]
      pos = pos - pos[0, :]
      pos = pos.at[0, 2].set(z_root)
      return pos
   
   def _get_quaternion(self, state: base.State):
      return state.x.rot
   
   def _get_velocity(self, state: base.State):
      return state.xd.vel
   
   def _get_angular_velocity(self, state: base.State):
      return state.xd.ang
   def _get_obs(self, state: base.State) -> Observation:

      obs = Observation(
         e_pos = self._get_egocentric_position(state),
         rot = self._get_quaternion(state),
         vel = self._get_velocity(state),
         ang = self._get_angular_velocity(state),
         ja = self._get_joint_angles(state),
         jv = self._get_joint_velocities(state),
         jr = self.normjr,
         d_goal = self._get_distance_to_goal(state)
      )
      return obs
   """

   
   



   '''
   def _get_goal(self):
      goal_pos = self.goal.pos
      if self.vel_goals:
         goal_vel = self.goal.vel
         return jp.concatenate(goal_pos, goal_vel, axis=-1)
      else:
         return goal_pos
   
   def _get_distance_to_goal(self, state: base.State):
      goal = self._get_goal
      if self.vel_goals:
         return goal - jp.concatenate(state.x.pos, state.xd.vel)
      else:
         return goal - state.x.pos'''
      

