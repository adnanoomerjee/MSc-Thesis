"""
Defines Observer and Observation classes
"""
from hct.envs.goal import Goal
from hct.envs.tools import world_to_relative, dist_quat, concatenate_attrs, pad
from hct.envs.old.env_tools import EnvTools

from brax.math import safe_norm
from brax import base

import jax
from jax import numpy as jp

from typing import Literal, Union


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
         return goal.xd.__sub__(sxd) * self.goal_xd_mask
            
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

   

   
   """
'''
def get_joint_angles(self, state: Union[base.State, Goal]):
   ja = self.scan_link_types(zero_if_non_hinge, 'q', 'l', state.q)
   ja = self\
      .vmap(in_axes = (None, 0, 0, 0, None, None))\
         .normalize_to_range(ja, self.jr[:,0], self.jr[:,1], -1, 1)
   return jp.expand_dims(ja, -1)

def get_joint_velocities(self, state: Union[base.State, Goal]):
   return jp.expand_dims(
      self.scan_link_types(zero_if_non_hinge, 'd', 'l', state.qd), -1
   )
'''
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
      

