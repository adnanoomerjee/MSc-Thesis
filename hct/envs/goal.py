
from hct.envs.env_tools import EnvTools
from hct.envs.tools import __mul__, q_spherical_to_quaternion, world_to_relative, random_ordered_subset

from brax import base, generalized
from brax.kinematics import forward
from brax.geometry import contact

import jax
from jax import numpy as jp

from flax import struct


@struct.dataclass
class Goal:
  """Target configuration for agent to acheive 

  Attributes:
      x_world: position in world frame
      x_rel: position in frame relative to parents
      xd_world: velocity in world frame 
      xd_rel: velocity in frame relative to parents
  """
  x_world: base.Transform
  x_rel: base.Transform
  xd_world: base.Motion
  xd_rel: base.Motion


















  
'''
class GoalConstructor(EnvTools):

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
      sys: base.System,
      configs: dict
  ):
    
    super().__init__(**configs, sys=sys)

  def sample_goal(self, rng: jp.ndarray, state: generalized.base.State):
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
    rng, rng1 = jax.random.split(rng)
    foot_contact_idx = random_ordered_subset(rng1, self.end_effector_idx)
    def sample(carry):
      rng, _ = carry
      rng, rng1 = jax.random.split(rng)
      g = jax.random.uniform(
          rng1, 
          shape=self.goal_size, 
          minval=self.goal_limit[:, 0], 
          maxval=self.goal_limit[:, 1]
        )
      goal = self.create_goal(g, state)
      return rng, goal
    def reject(carry):
      rng, goal = carry
      cpos_foot_z = contact(self.sys, goal.x).pos[foot_contact_idx, 2]
      z = goal.x.pos[:,2]
      polar = goal.bpg[6]
      cond = jp.any(z < self.goal_z_cond[0]) | \
        jp.any(z > self.goal_z_cond[1]) | \
        (polar > self.goal_polar_cond) | \
        jp.any(cpos_foot_z > self.goal_contact_cond[1]) | \
        jp.any(cpos_foot_z < self.goal_contact_cond[0])
      return cond
    init_g = jax.random.uniform(
        rng, 
        shape=self.goal_size, 
        minval=-1, 
        maxval=1
    )
    init_val = rng, self.create_goal(init_g, state)
    goal = jax.lax.while_loop(reject, sample, init_val)[1]
    return goal

  def create_goal(self, g: jp.ndarray, state: base.State, sys: base.System) -> Goal:
    """
    Creates a goal state 

    Args:
      g: jp.ndarray normalised goal
      state: environment state

    Returns:
      Goal: goal state
    """
    q = q_spherical_to_quaternion(g[:self.dof], state, sys)
    qd = g[-self.dof:]
    x_world, xd_world = forward(self.sys, q, qd)
    x_world, xd_world = __mul__(x_world, self.goal_x_mask), __mul__(xd_world, self.goal_xd_mask)
    x_rel, xd_rel = world_to_relative(x_world, sys), world_to_relative(xd_world, sys)
    x_rel, xd_rel = __mul__(x_rel, self.goal_x_mask), __mul__(xd_rel, self.goal_xd_mask)
    return Goal(x_world, x_rel, xd_world, xd_rel)

'''




'''
def q_to_bpgq(self, q: jp.ndarray, state: base.State):
  def truefunc():
    def scanfunc(link_type, feature):
      """Converts backprojected q goal to q state"""
      def ffunc(feature):
        pos = feature[0:3] - state.x.pos[0] # root pos from world to state egocentric
        rot = quaternion_to_spherical(feature[3:7])
        return jp.concatenate([pos,rot])
      def nonffunc(feature):
        return feature
      return jax.lax.cond(link_type != 'f', nonffunc, ffunc, feature)
    return self.scan_link_types(scanfunc, 'q', 'd', q)
  def falsefunc():
    return jp.zeros(self.goal_size)
  return jax.lax.cond(self.q_goals, truefunc, falsefunc)
'''
    


  



        





'''
    def if_q_qd_goals():
      def if_root_qd_goals():
        q = q()
        qd = qd()*self.qd_mask
        x, xd = forward(self.sys, q, qd)
        return Goal(self.q_goals, self.qd_goals, g, q, qd, x, xd)
      def if_full_qd_goals():
        q = q()
        qd = qd()
        x, xd = forward(self.sys, q, qd)
        return Goal(self.q_goals, self.qd_goals, g, q, qd, x, xd)
  
    def if_not_q_qd_goals():
      def if_q_goals():
        q = q()
        qd = jp.zeros(self.dof)
        x, xd = forward(self.sys,q, qd)
        return Goal(self.q_goals, self.qd_goals, g, q, qd, x, xd)
      def if_qd_goals():
        q = jp.zeros(self.q_size)
        qd = qd()
        root_motion = base.Motion(qd[0:3], qd[3:6])
        x = base.Transform.zero((self.num_links,))
        xd = base.Motion.zero((self.num_links,)).index_set(0, root_motion)
      return jax.lax.cond(self.q_goals, if_q_goals, if_qd_goals)

    g_obs = jax.lax.cond(
         self.q_goals, if_q_goals, if_not_q_goals
      )
    
    def if_q_goals():
      q = q() * self.q_mask
      qd = qd() * self.qd_mask
      x, xd = forward(self.sys, q, qd)
      xd = x.__mul__(self.x_mask)
      xd = xd.__mul__(self.xd_mask)
      return Goal(self.q_goals, self.qd_goals, g, q, qd, x, xd)
    
    def if_not_q_goals():
      q = self.q_mask
      qd = qd() * self.qd_mask
      xd = base.Motion(qd[0:3], qd[3:6])\
        .concatenate(base.Motion(jp.tile(jp.zeros(self.num_links))))

    def q_goals_func(bpg_q):
      q = self.scan_link_types(self.bpg_q_to_q, 'd', 'q', bpg_q)
      root_pos, root_rot = (q[0:3], q[3:7])
      root_t = base.Transform(root_pos, root_rot)
      root_t = egocentric_to_world(root_t, state.x)
      return q.at[0:7].set(jp.concatenate([root_t.pos, root_t.rot], axis=-1))
    def non_q_goals_func(bpg_q):
      return jp.zeros(self.q_size).at[3].set(1)
    
    bpg = self.backproj(g)
    q = jax.lax.cond(
      self.q_goals, q_goals_func, non_q_goals_func, bpg[self.dof:])
    qd = jax.lax.select(self.qd_goals, bpg[-self.dof:], jp.zeros(self.dof))
    x, xd = forward(self.sys, q, qd)

    return Goal(self.q_goals, self.qd_goals, g, q, qd, x, xd)'''
    