from brax import base, scan

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
         joint_obs: bool = False,
         goal_obs: Literal['concatenate', 'node'] = 'concatenate',
         q_goals: bool = True,
         qd_goals: Literal[None, 'root', 'full'] = None, 
         goal_x_range: Tuple[base.Transform, base.Transform] = None,
         goal_xd_range: Tuple[base.Motion, base.Motion] = None,
         goal_root_pos_range: jp.ndarray = jp.array([[-3,3], [-3,3], [-0.25, 0.6]]),
         goal_root_rot_range: jp.ndarray = jp.array([[-jp.pi,jp.pi], [0, jp.pi], [-jp.pi, jp.pi]]),
         goal_qd_limit: Optional[jp.ndarray] = None
   ):
      # Agent attributes
      self.sys = sys
      self.dof_limits = jp.array(self.sys.dof.limit).T
      self.dof = self.dof_limits.shape[0]
      self.jr = self._get_node_jr()
      self.normjr = self._get_node_normalised_jr()
      self.q_size = sys.q_size()
      self.qd_size = sys.q_size()
      self.num_links = sys.num_links()
      self.joint_obs = joint_obs
      self.obs_width = 15 if joint_obs else 13

      # Goal attributes
      self.goal_nodes = True if goal_obs == 'node' else False
      self.q_goals = q_goals
      self.qd_goals = False if qd_goals is None else True
      self.root_qd_goals = True if qd_goals == 'root' else False
      self.full_qd_goals = True if qd_goals == 'full' else False
      self.goal_size = (self.dof*2,)

      assert q_goals or self.qd_goals, "Must specify at least one goal type"
      assert not (self.qd_goals and goal_qd_limit is None), "Must specify g_qd limit"
      assert not (self.qd_goals and goal_qd_limit.shape[0] != self.dof), "Must specify correctly configured g_qd limits"

      if self.q_goals and self.full_qd_goals:
         self.goal_g_mask = 1
         self.goal_q_mask = 1
         self.goal_qd_mask = 1
         self.goal_x_mask = 1
         self.goal_xd_mask = 1
         self.optimal_goal_obs = jp.zeros((self.num_links, 15)).at[:,3].set(1)
      elif self.q_goals and self.root_qd_goals:
         self.goal_g_mask = jp.zeros(self.goal_size).at[:self.dof+6].set(1.0)
         self.goal_q_mask = 1
         self.goal_qd_mask = jp.zeros(self.qd_size).at[0:6].set(1.0)
         self.goal_x_mask = 1
         self.goal_xd_mask = jp.zeros((self.num_links,3)).at[0].set(1.0)
         self.optimal_goal_obs = jp.zeros((self.num_links, 14)).at[:,3].set(1)
      elif self.q_goals:
         self.goal_g_mask = jp.zeros(self.goal_size).at[self.dof].set(1.0)
         self.goal_q_mask = 1
         self.goal_qd_mask = 0
         self.goal_x_mask = 1
         self.goal_xd_mask = 0
         self.optimal_goal_obs = jp.zeros((self.num_links, 8)).at[:,3].set(1)
      elif self.full_qd_goals:
         self.goal_g_mask = jp.ones(self.goal_size).at[:self.dof].set(0.0)
         self.goal_q_mask = 0
         self.goal_qd_mask = 1
         self.goal_x_mask = 0
         self.goal_xd_mask = jp.zeros((self.num_links,3)).at[0].set(1.0)
         self.optimal_goal_obs = jp.zeros((self.num_links, 7))
      else:
         assert self.q_goals, "Cannot only specify root_qd_goal"

      if self.goal_nodes:
         self.optimal_goal_obs = jp.concatenate([pad(self.optimal_goal_obs, self.obs_width), jp.ones(self.num_links, 1)], axis=-1)
      
      self.goal_q_limit = self.dof_limits.at[0:3].set(goal_root_pos_range).at[3:6].set(goal_root_rot_range) if q_goals else jp.zeros((self.dof,2))
      self.goal_qd_limit = goal_qd_limit if self.qd_goals else jp.zeros((self.dof,2))
      self.goal_limit = jp.concatenate([self.goal_q_limit, self.goal_qd_limit])

      # Goal sampling parameters
      if env_name == 'ant':
         self.end_effector_idx = [2,4,6,8]
         self.goal_z_cond = jp.array([0.078, 1.3])
         self.goal_polar_cond = jp.pi/12
         self.goal_contact_cond = [-0.003, 0.01]


   def __mul__(self, a, b: Any) -> Any:
      return jax.tree_map(lambda x, y: x * y, a, b)
   
   def normalize_to_range(
         self,
         value, 
         min_value, 
         max_value, 
         a=-1, 
         b=1
      ):
      def func(value):
         return a + ((value - min_value) * (b - a)) / (max_value - min_value)
      return jax.lax.cond(max_value == min_value, func, zero, value)
   
   def unnormalize_to_range(
         self,
         normalized_value, 
         min_value, 
         max_value, 
         a=-1, 
         b=1
      ):
      return min_value + ((normalized_value - a) * (max_value - min_value)) / (b - a)

   def scan_link_types(self, link_type_condition, in_types, out_types, *args):
      """Scan a function over System link type ranges"""
      return scan.link_types(self.sys, link_type_condition, in_types, out_types, *args)
      
   def _get_node_jr(self):
      """Returns joint range of each hinge joint for each node
      All non hinge nodes set to zero

      Returns:
         jr: (num_links, 2) joint ranges
      """
      dof = self.dof_limits.T
      return self\
         .vmap(in_axes=(None, None, None, None, 0))\
            .scan_link_types(zero_if_non_hinge, 'd', 'l', dof).T
   
   def _get_node_normalised_jr(self):
      '''Returns normalised joint ranges of each hinge joint for each node
      All non hinge nodes set to zero

      Returns:
         jr: (num_links, 2) joint ranges, normalised
      '''
      dof = jp.array(self.sys.dof.limit)
      dof = self.normalize_to_range(dof, -jp.pi, jp.pi)
      return self\
         .vmap(in_axes=(None, None, None, None, 0))\
            .scan_link_types(zero_if_non_hinge, 'd', 'l', dof).T
   
   
def concatenate_attrs(obj):
   return jp.concatenate([v for k, v in vars(obj).items()], axis=-1)

def scan_concatenate_attrs(carry, obj):
   return jp.concatenate([carry, concatenate_attrs(obj)], axis=-1)

def zero(x):
   return jp.zeros_like(x)

def zero_if_non_hinge(link_type, feature):
   """Sets joint ranges to zero if not a hinge joint
   Intended to be passed as argument to scan_link_types
   """
   if link_type == '1':
      return feature
   else:
      return jp.zeros(1)

def pad(tensor, length, axis = -1):
    padding_length = length - tensor.shape[axis]
    padding_tensor = jp.zeros((tensor.shape[axis-1], padding_length))
    return jp.concatenate([tensor, padding_tensor], axis = axis)