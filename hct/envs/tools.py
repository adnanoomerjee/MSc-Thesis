import time

from hct.envs.goal import Goal

from brax.math import *
from brax.base import Motion, State, System, Transform
from brax.scan import link_types
from brax.kinematics import inverse, world_to_joint
from brax.envs import Env
import brax

import jax
import jax.numpy as jp
from typing import Any, Callable, Optional

import types, functools
from multipledispatch import dispatch

def mul(a: Any, b: Any) -> Any:
    return jax.tree_map(lambda x, y: x * y, a, b)

def concatenate_attrs(obj):
   return jp.concatenate([v for k, v in vars(obj).items()], axis=-1)

def pad(tensor, length, axis = -1):
    padding_length = length - tensor.shape[axis]
    padding_tensor = jp.zeros((tensor.shape[axis-1], padding_length))
    return jp.concatenate([tensor, padding_tensor], axis = axis)

def normalize_to_range(
        value, 
        min_value, 
        max_value, 
        a=-1, 
        b=1
    ):        
    nonzero = jp.where(jp.abs(max_value - min_value) > 1e-4, x=1.0, y=0.0)
    max_value = max_value + (1 - nonzero)

    return nonzero * (a + ((value - min_value) * (b - a))) / (max_value - min_value)

def unnormalize_to_range(
        normalized_value, 
        min_value, 
        max_value, 
        a=-1, 
        b=1
    ):
    return min_value + ((normalized_value - a) * (max_value - min_value)) / (b - a)

def zero(x):
   return jp.zeros_like(x)

def spherical_to_quaternion(v: jp.ndarray) -> jp.ndarray:
    r, theta, phi = v
    # Convert spherical coordinates to Cartesian coordinates to represent the axis of rotation
    x = jp.sin(theta) * jp.cos(phi)
    y = jp.sin(theta) * jp.sin(phi)
    z = jp.cos(theta)
    axis = jp.array([x, y, z])
    # The radial distance is used as the rotation angle
    angle = r
    quat = quat_rot_axis(axis, angle)
    return quat

def spherical_to_unit_cartesian(v: jp.ndarray) -> jp.ndarray:
    r, theta, phi = v
    x = jp.sin(theta) * jp.cos(phi)
    y = jp.sin(theta) * jp.sin(phi)
    z = jp.cos(theta)
    return jp.array([x, y, z]), r

def quaternion_to_spherical(quat: jp.ndarray) -> jp.ndarray:
    w = quat[0]
    xyz = quat[1:]
    # Convert quaternion to cartesian axis-angle
    v, _ = normalize(xyz)
    angle = 2*safe_arccos(w)
    # convert cartesian axis-angle to spherical
    r = jax.lax.select(angle <= jp.pi, angle, angle - 2*jp.pi)
    theta = safe_arccos(v[2]) # inclination angle
    theta = jp.where(jp.abs(r) == 0.0, theta*0, theta)
    phi = jp.arctan2(v[1], v[0]) # azimuthal angle
    return jp.array([r, theta, phi])

def q_spherical_to_quaternion(g: jp.ndarray, state: State, sys: System):
    def scanfunc(link_type, feature):
      """Converts backprojected q goal to q state"""
      if link_type == 'f':
        pos = feature[0:3] + state.x.pos[0] # root pos from state egocentric to world
        z0 = jp.array([0, 0, 1])
        new_z, angle = spherical_to_unit_cartesian(feature[3:6])
        rot = from_to(z0, new_z)
        rot = quat_mul(quat_rot_axis(new_z, angle), rot)
        #rot = spherical_to_quaternion(feature[3:6])
        return jp.concatenate([pos, rot], axis=-1)
      else:
        return feature
    return link_types(sys, scanfunc, 'd', 'q', g)


def world_to_relative(state, env):
    
    link_parents = jp.array(env.link_parents)
    link_parents = jp.where(link_parents == -1, 0, link_parents)

    x = state.x
    xd = state.xd
    root_x = x.take(0)
    root_xd = xd.take(0)

    x_link_parents = x.take(link_parents)
    xd_link_parents = xd.take(link_parents)

    x_rel = x.vmap(in_axes=(0,0)).to_local(x_link_parents).index_set(0, root_x)
    
    quat = x_link_parents.rot
    xd_rel = jax.tree_map(lambda x: jax.vmap(inv_rotate)(x, quat), xd.__sub__(xd_link_parents)).index_set(0, root_xd)

    return x_rel, xd_rel

def relative_to_world(state, env):
    
    link_parents = jp.array(env.link_parents)
    link_parents = jp.where(link_parents == -1, 0, link_parents)

    x = state.x
    xd = state.xd
    root_x = x.take(0)
    root_xd = xd.take(0)

    x_link_parents = x.take(link_parents)
    xd_link_parents = xd.take(link_parents)

    x = x.vmap(in_axes=(0,0)).do(x_link_parents).index_set(0, root_x)
    y = x.vmap(in_axes=(0, None)).do(root_x)
    x = x.index_set(jp.array([2,4,6,8]), jax.tree_map(lambda x: x[jp.array([2,4,6,8])], y))
    
    quat = x_link_parents.rot
    xd = jax.tree_map(lambda x: jax.vmap(rotate)(x, quat), xd.__add__(xd_link_parents)).index_set(0, root_xd)
    yd = jax.tree_map(lambda x: jax.vmap(rotate)(x, quat), xd.__add__(root_xd))
    xd = xd.index_set(jp.array([2,4,6,8]), jax.tree_map(lambda x: x[jp.array([2,4,6,8])], yd))

    return x, xd


def world_to_egocentric(state):

    x = state.x
    xd = state.xd
    root_x = x.take(0)
    root_xd = xd.take(0)

    x_rel = x.vmap(in_axes=(0, None)).to_local(root_x).index_set(0, root_x)
    xd_rel = jax.tree_map(lambda x: jax.vmap(inv_rotate, in_axes=(0, None))(x, root_x.rot), xd.__sub__(root_xd)).index_set(0, root_xd)

    return State(state.q, state. qd, x_rel, xd_rel, None)

    
def egocentric_to_world(state):
    return None

def dist_quat(quat1, quat2):
    """Cosine angle distance between quaternions"""
    cosine = jp.minimum(jp.dot(quat1, quat2), jp.dot(quat1, -quat2))
    return jp.expand_dims(jp.abs(cosine - 1), -1)

def random_ordered_subset(rng, idx: jp.ndarray):
    rng, rng1, rng2 = jax.random.split(rng, 3)
    n = jax.random.choice(rng2, jp.array(range(0, len(idx)+1)))
    subset = jax.random.choice(rng1, jp.array(idx), shape=(n,), replace = False)
    return jp.sort(subset)

def shortest_angle_diff(angle1, angle2):
    diff = angle2 - angle1
    diff = (diff + jp.pi) % (2 * jp.pi) - jp.pi
    return diff

def minmax_angles(angles, goal_root_rot_range = None):

    x = jp.cos(angles)
    y = jp.sin(angles)

    jp.arctan2(y, x)

    # Compute the center of all angles
    center = jp.arctan2(jp.sum(jp.sin(angles), axis=0), jp.sum(jp.cos(angles), axis=0))

    # Subtract the center from all angles
    difference_from_center = shortest_angle_diff(angles, center)
    argmin = jp.argmin(difference_from_center, axis=0)
    argmax = jp.argmax(difference_from_center, axis=0)
    
    minimum = jp.take_along_axis(angles, argmin[None, ...], axis=0)
    maximum = jp.take_along_axis(angles, argmax[None, ...], axis=0)

    if goal_root_rot_range is not None:
        minimum = minimum.at[0].set(goal_root_rot_range[:,0])
        maximum = maximum.at[0].set(goal_root_rot_range[:,1])

    return minimum, maximum

def timeit(fn: Callable, *args):
    start_time = time.time()
    fn_output = jax.block_until_ready(fn(*args))
    end_time = time.time()
    return fn_output, end_time - start_time

def goal_to_state(goal: Goal, sys: System, state):
    goal = brax.positional.base.State(
        q=goal.q,
        qd=goal.qd,
        x=goal.x,
        xd=goal.xd,
        contact=None,
        x_i=state.x_i,
        xd_i=state.xd_i, 
        j=state.j, 
        jd=state.jd,
        a_p=state.a_p, 
        a_c=state.a_c, 
        mass=state.mass,
    )
    return goal


def dist(env: Env, x1, xd1, x2, xd2, rot_dist=False, root_dist=False, importance = None):

    if importance is None:
        importance = 1
    
    dpos = 2*(x1.pos - x2.pos)#/env.minmax['pos']
    drot = jp.zeros((env.num_nodes, 1)).at[0,0].set(1) * jax.vmap(dist_quat)(x1.rot, x2.rot)#/(env.minmax['rot'])
    dx = jp.concatenate([dpos, drot], axis=-1) * env.goal_x_mask

    dvel = jp.expand_dims(safe_norm(xd1.vel, axis=-1) - safe_norm(xd2.vel, axis=-1), axis=-1)#/env.minmax['vel']
    dang = 0 * (xd1.ang - xd2.ang)#/env.minmax['ang']
    dxd = jp.concatenate([dvel, dang], axis=-1) * env.goal_xd_mask

    s_minus_g_sq = jp.square(jp.concatenate([dx, dxd], axis=-1) * importance) 
    
    if env.obs_mask is not None:
        s_minus_g_sq = s_minus_g_sq * env.obs_mask

    s_minus_g_sq = jp.where(jp.isnan(s_minus_g_sq), x=0.0, y=s_minus_g_sq)
    root_dist = jp.sqrt(jp.sum(s_minus_g_sq[0])) if root_dist else None
    dist = jp.sqrt(jp.sum(s_minus_g_sq)) #jp.clip(jp.sqrt(jp.sum(s_minus_g_sq)), 0, jp.sqrt(jp.sum(2 * importance))) 

    return dist, root_dist


def max_sq_dist_nodes(env: Env, x1, xd1, x2, xd2, root_dist=False):

    dpos = jp.square(x1.pos - x2.pos)
    drot = jp.square(jax.vmap(dist_quat)(x1.rot, x2.rot))
    dx = jp.concatenate([dpos, drot], axis=-1) * env.goal_x_mask

    dvel = jp.square(xd1.vel - xd2.vel) 
    dang = jp.square(xd1.ang - xd2.ang) 
    dxd = jp.concatenate([dvel, dang], axis=-1) * env.goal_xd_mask

    s_minus_g_sq = jp.concatenate([dx, dxd], axis=-1)
    
    if env.obs_mask is not None:
        s_minus_g_sq = s_minus_g_sq * env.obs_mask
    
    sq_dists = jp.clip(jp.sum(s_minus_g_sq, axis=1, keepdims=True), 0, jp.sqrt(4 * 9))

    return sq_dists


def slerp(q1, q2, t):
    """Perform SLERP between two quaternions q1, q2 with interpolant t."""

    # Compute the cosine of the angle between the two vectors.
    dot = jp.dot(q1, q2)

    # If the dot product is negative, the quaternions
    # have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    q2 = jp.where(dot < 0.0, -q2, q2)
    dot = jp.where(dot < 0.0, -dot, dot)

    DOT_THRESHOLD = 0.9995

    def f():
    
        # If the inputs are too close for comfort, linearly interpolate
        # and normalize the result.
        result = q1 + t*(q2 - q1)
        return result / jp.linalg.norm(result)
    
    def g():
            # Since dot is in range [0, DOT_THRESHOLD], acos is safe
        theta_0 = safe_arccos(dot)  # theta_0 = angle between input vectors
        theta = theta_0 * t  # theta = angle between v0 and result
        sin_theta = jp.sin(theta)  # compute this value only once
        sin_theta_0 = jp.sin(theta_0)  # compute this value only once

        s0 = jp.cos(theta) - dot * sin_theta / sin_theta_0  # == sin(theta_0 - theta) / sin(theta_0)
        s1 = sin_theta / sin_theta_0
        return s0 * q1 + s1 * q2
    
    return jax.lax.cond(dot > DOT_THRESHOLD, f, g)






'''
@world_to_relative.register(Transform)
def _(t, sys: System, mask_root: bool = False) -> Transform:
    """Converts a transform in world frame to transform in frame relative to parent link.

    Args:
        t: transform in world frame

    Returns:
        t': transform in relative frame
    """
    link_parents = jp.array(sys.link_parents)
    link_parents = jp.where(link_parents == -1, 0, link_parents)
    root = t.take(0)
    if mask_root:
        root_mask = Transform(jp.array([0.0, 0.0, 1.0]), jp.array([1.0, 1.0, 1.0, 1.0]))
        root = mul(root, root_mask)
    r = t.take(link_parents)
    return t.vmap(in_axes=(0,0)).to_local(r).index_set(0, root)

@world_to_relative.register(Motion)
def _(m: Motion, sys: System, mask_root: bool = False) -> Motion:
    """Converts a transform in world frame to transform in frame relative to parent link.

    Args:
        t: transform in world frame

    Returns:
        t': transform in relative frame
    """
    link_parents = jp.array(sys.link_parents)
    link_parents = jp.where(link_parents == -1, 0, link_parents)
    root = m.take(0)
    r = m.take(link_parents)
    return m.__sub__(r).index_set(0,root)
'''













'''


def world_to_link(t, x) -> Transform:
    """Converts a transform in world frame to transform in link frame.

    Args:
        t: transform in world frame
        x: world frame transform of agent

    Returns:
        t': transform in link frame
    """
    return t.vmap(in_axes=(0,0)).to_local(x)
def egocentric_to_link(t, x) -> Transform:
    """Converts a transform in egocentric frame to transform in link frame.

    Args:
        t: transform in egocentric frame
        x: world frame transform of agent

    Returns:
        t': transform in link frame
    """
    t = egocentric_to_world(t, x)
    t = world_to_link(t, x)
    return t

'''
'''
@dispatch(Motion, Motion)
def world_to_link(m, xd) -> Motion:
    """Converts a Motion in world frame to Motion in link frame.

    Args:
        m: Motion in world frame
        xd: world frame transform of agent

    Returns:
        m': transform in link frame
    """
    return m.vmap(in_axes=(0,0)).to_local(x)

@dispatch(Motion, Motion)
def world_to_egocentric(m, xd) -> Transform:
    """Converts a motion in world frame to motion in egocentric frame.

    Args:
        m: motion in world frame
        xd: world frame motion of agent

    Returns:
        m': motion in egocentric frame
    """
    rpos = x.pos[0]
    rrot = x.rot[0]
    rx = Transform.create(rpos,rrot)
    return t.vmap(in_axes=(0,None)).to_local(rx)
'''