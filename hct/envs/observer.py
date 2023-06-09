"""
Defines Observer and Observation classes
"""

from brax import base, scan, math
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import world_to_joint
from etils import epath
import jax
from jax import numpy as jp
from flax import struct
from typing import Any, Dict, List, Tuple, Union, Optional

class Observer(base.Base):
   """API for observing a brax system"""

   def __init__(
         self,
         sys: base.System,
         feature_subset: Tuple[int] = None
   ):
      self.sys = sys
      self.obs = None

   def get_obs(self, state: base.State):
      self.obs = self._get_obs(state)
      obs = self.obs.format_for_input()
      return obs 

   def _get_obs(self, state: base.State):
      obs = Observation(
         e_pos = self._get_egocentric_position(state),
         rot = self._get_quaternion(state),
         vel = self._get_velocity(state),
         ang = self._get_angular_velocity(state),
         ja = self._get_joint_angles(state),
         jv = self._get_joint_velocities(state),
         jr = self._get_joint_ranges(),
      )
      return obs

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

   def _get_joint_angles(self, state: base.State):
      '''Returns joint angles of each joint'''
      return self._scan_link_types('q', state.q)

   def _get_joint_velocities(self, state: base.State):
      """Returns joint velocity of each joint"""
      return self._scan_link_types('qd', state.qd)

   def _get_joint_ranges(self):
      """Returns joint range of each joint"""
      dof = jp.array(self.sys.dof.limit)
      return self.vmap(in_axes=(None, None, None, None, 0))._scan_link_types('qd', dof)
   
   def _scan_link_types(self, in_types, feature):
      """Scan a function over System link type ranges"""
      return scan.link_types(self.sys, self.link_type_condition, in_types, 'l', feature)

   @staticmethod
   def _link_type_condition(link_type, feature):
      """Sets joint ranges to zero if not a hinge joint"""
      if link_type == '1':
         return feature
      else:
         return jp.zeros(1)
    

@struct.dataclass
class Observation(base.Base):

   """Dynamic observation that changes every pipeline step.

   Attributes:
      e_pos: (num_links, 3) position transform of the egocentric frame, with z coordinate of root in world frame
      rot: (num_links, 4) quaternion rotation in world frame
      vel: (num_links, 3) link velocity in world frame
      ang: (num_links, 3) link angular velocity in world frame
      ja: (num_links,) joint angles
      jv: (num_links,) joint velocities
      jr: (num_links, 2) joint ranges

   """

   e_pos: jp.ndarray
   rot: jp.ndarray
   vel: jp.ndarray
   ang: jp.ndarray
   ja: jp.ndarray
   jv: jp.ndarray
   jr: jp.ndarray

   def format_for_input(self):
      """
      Concatenates the observation attributes into a tensor for input
      into transformer model

      Returns:
         2D tensor, each row corresponding to agent node with features.
      """
      return jp.concatenate([v for k, v in vars(self).items()], axis=-1)