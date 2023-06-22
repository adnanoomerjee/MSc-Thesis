
from brax import base, scan
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import world_to_joint

from etils import epath

import jax
from jax import numpy as jp

from flax import struct

from typing import Any, Dict, List, Tuple, Union, Optional
from brax.kinematics import forward, world_to_joint, inverse

from hct.envs.env_tools import Base
from hct.envs import math

class GoalConstructor(Base):

  """
  TODO

  Add attribute g to Goal class, representing joint configuration mapped to [-1, 1]

  Add functionality for sampling body rotation 

  Add function for generating goal from upper level distribution

  Add optionality for sampling velocity

  Add optionality for body rotations

  Add optionality for further restriction of position ranges
    Restrict such that entire body is above z = 0

  """
  def __init__(
      self, 
      num_nodes: int,
      max_dof: int,
      sys: base.System,
      q_goals: bool = True,
      qd_goals: bool = False,
      root_pos_range: jp.ndarray = jp.array([
        [-3,3],
        [-3,3], 
        [0.3,1]]),
      root_rot_range: jp.ndarray = jp.array([
        [-jp.pi,jp.pi],
        [0,jp.pi], 
        [0,2*jp.pi]]),
      qd_limit: Optional[jp.ndarray] = None
      ):
    
    super().__init__(sys=sys)

    self._q_goals = q_goals
    self._qd_goals = qd_goals
    self._joint_limits = jp.array(self.sys.dof.limit).T
    self._dof = self._joint_limits.shape[0]
    
    self._q_limit = self._dof_limits.at[0:3].set(root_pos_range).at[3:6].set(root_rot_range)

    if qd_goals:
      self._qd_limit = qd_limit
    else:
      self._qd_limit = jp.zeros((sys._dof, 2))

    self._g_size = (num_nodes, (int(q_goals) + int(qd_goals))*max_dof)

  def sample_goal(self, rng: jp.ndarray):
    """
    Samples joint configuration q from predefined q range.

    Returns:
    Target torso XYZ position, egocentric position of limbs.
    """
    
    rng, rng1, rng2 = jax.random.split(rng, 3)

    g = jax.random.uniform(
      rng1, 
      shape=(self._g_size,), 
      minval=-1, 
      maxval=1
    )

    q_g, qd_g = self.backproj(g)
    x_g, xd_g = forward(self.sys, q_g, qd_g)
   
    pos = sample.x.pos
    root_pos = pos[0,:]
    e_pos = pos - root_pos
    e_pos = pos.at[0].set[root_pos]

    goal = Goal(pos=e_pos)

    return goal
  
  def _get_q_g(self, g: jp.ndarray):
    if self._q_goals:
      return 
    
  def backproj(self, g: jp.ndarray):

    g = jax.vmap(math.unnormalize_to_range)

    if self._q_goals:
      root_pos = \
        self.\
          vmap(0, 0, 0, None, None)\
            .unnormalize_to_range(
              g[0:3],
              min_value=self._q_limit[:3, 0],
              max_value=self._q_limit[:3, 1]
            )
      root_rot = self._get_quaternion(g[3:6])
      ja = \
        self.\
          vmap(0, 0, 0, None, None)\
            .unnormalize_to_range(
              g[6:],
              min_value=self._q_limit[6:, 0],
              max_value=self._q_limit[6:, 1]
            )
      
      q_g = jp.concatenate([root_pos, root_rot, ja])
    else:
      q_g = jp.zeros(self.q_size)

    if self._qd_goals:
      qd_g = \
        self.\
          vmap(0, 0, 0, None, None)\
            .unnormalize_to_range(
              g[self._dof:],
              min_value=self._qd_limit[:, 0],
              max_value=self._qd_limit[:, 1]
            )
    else:
      qd_g = jp.zeros(self.qd_size)

    return q_g, qd_g
      

  def _get_quaternion(self, X):

    X = (X + 1) / 2
    θ = 2*jp.pi*X[1:]
    s = jp.sin(θ)
    c = jp.cos(θ)
    r = jp.sqrt([1 - X[0], X[0]])
    sr = s*r
    cr = c*r

    rot = jp.array(
      [
        cr[1],
        sr[0],
        cr[0],
        sr[1]
      ]
    )

    return rot
  
  def inv_subgroup_algorithm(self, rot):

    r1 = rot[0]**2 + rot[3]**2

    X0 = r1

    θ0 = jp.arctan2(rot[2],rot[1]) + jp.pi
    θ1 = jp.arctan2(rot[0],rot[3]) + jp.pi
    
    X1 = θ0 / (2*jp.pi)
    X2 = θ1 / (2*jp.pi)
    
    X = jp.array([X0, X1, X2])
    
    X = 2*X-1
    return X


      
  

@struct.dataclass
class Goal(base.Base):
  """Target configuration for agent to acheive 

  Attributes:
    g: (dof_size,) goal dof features, mapped to [-1,1]
    pos: (num_links, 3) position Transform, with root in world frame and limbs in egocentric frame
    vel: (num_links, 3) link velocity in world frame

  """
  g: jp.ndarray
  q_g: Optional[jp.ndarray]
  qd_g: Optional[jp.ndarray]
  x_g: Optional[base.Transform]
  xd_g: Optional[base.Motion]




