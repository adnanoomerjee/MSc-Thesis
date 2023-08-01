from brax import base, scan

from hct.envs.tools import normalize_to_range

import jax
from jax import numpy as jp
from jaxlib.xla_extension import ArrayImpl

from flax import struct

from typing import Any, Literal, Optional, Tuple


class EnvTools(base.Base):
   """
   TODO: account for humanoid (end effector ID)
   """
   
   def __init__(
         self,
         sys: base.System,
         env_name: Literal['ant', 'humanoid'] = 'ant', 
         goal_obs: Literal['concatenate', 'node'] = 'concatenate',
         position_goals: bool = True,
         velocity_goals: Literal[None, 'root', 'full'] = None, 
         goal_x_range: Tuple[base.Transform, base.Transform] = None,
         goal_xd_range: Tuple[base.Motion, base.Motion] = None,
         goalsampler_root_pos_range: jp.ndarray = jp.array([[-3,3], [-3,3], [-0.25, 0.6]]),
         goalsampler_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi], [-jp.pi, jp.pi]]),
         goalsampler_qd_limit: Optional[jp.ndarray] = None
   ):

      # Agent attributes
      self.sys = sys
      self.dof = jp.array(self.sys.dof.limit).T.shape[0]
      self.num_links = sys.num_links()
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

      # Goal sampling parameters
      if env_name == 'ant':
         self.end_effector_idx = [2,4,6,8]
         self.goal_z_cond = jp.array([0.078, 1.3])
         self.goal_polar_cond = jp.pi/12
         self.goal_contact_cond = [-0.003, 0.01]

      goalsampler_q_limit = jp.array(self.sys.dof.limit).T.at[0:3].set(goalsampler_root_pos_range).at[3:6].set(goalsampler_root_rot_range)
      self.goalsampler_limit = jp.concatenate([goalsampler_q_limit, goalsampler_qd_limit])



#   def scan_link_types(self, link_type_condition, in_types, out_types, *args):
"""Scan a function over System link type ranges"""
#      return scan.link_types(self.sys, link_type_condition, in_types, out_types, *args)

'''
def scan_concatenate_attrs(carry, obj):
   return jp.concatenate([carry, concatenate_attrs(obj)], axis=-1)

def zero_if_non_hinge(link_type, feature):
   if link_type == '1':
      return feature
   else:
      return jp.zeros(1)
'''
#def _get_node_jr(self):
"""Returns joint range of each hinge joint for each node
All non hinge nodes set to zero

Returns:
   jr: (num_links, 2) joint ranges
"""
#   dof = self.dof_limits.T
#   return self\
#      .vmap(in_axes=(None, None, None, None, 0))\
#         .scan_link_types(zero_if_non_hinge, 'd', 'l', dof).T

#def _get_node_normalised_jr(self):
'''Returns normalised joint ranges of each hinge joint for each node
All non hinge nodes set to zero

Returns:
   jr: (num_links, 2) joint ranges, normalised
'''
#   dof = jp.array(self.sys.dof.limit)
#   dof = normalize_to_range(dof, -jp.pi, jp.pi)
#   return self\
#      .vmap(in_axes=(None, None, None, None, 0))\
#         .scan_link_types(zero_if_non_hinge, 'd', 'l', dof).T