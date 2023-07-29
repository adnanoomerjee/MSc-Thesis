import time

from hct.envs.goal import Goal

from brax.math import *
from brax.base import Motion, State, System, Transform
from brax.scan import link_types
from brax.kinematics import inverse, world_to_joint

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
    def func(value):
        return a + ((value - min_value) * (b - a)) / (max_value - min_value)
    return jax.lax.cond(max_value == min_value, func, zero, value)

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

def world_to_egocentric(t: Transform) -> Transform:
    """Converts a transform in world frame to transform in egocentric frame.

    Args:
        t: transform in world frame

    Returns:
        t': transform in egocentric frame
    """
    r = t.take(0)
    return t.vmap(in_axes=(0,None)).to_local(r)

def egocentric_to_world(t: Transform) -> Transform:
    """Converts a transform in egocentric frame to transform in world frame.

    Args:
        t: transform in egocentric frame

    Returns:
        t': transform in world frame
    """
    r = t.take(0)
    return r.vmap(in_axes=(None,0)).do(t)

@functools.singledispatch
def world_to_relative(t, sys: System, mask_root: bool = False):
    del t, sys
    return NotImplemented

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

@jax.vmap
def dist_quat(quat1, quat2):
    """Cosine angle distance between quaternions"""
    return jp.expand_dims(jp.abs(jp.dot(quat1, quat2) - 1), -1)

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

def goal_to_state(goal: Goal, sys: System):
    x = goal.x_world
    xd = goal.xd_world
    j, jd, _, _ = world_to_joint(sys, x, xd)
    q, qd = inverse(sys, j, jd)
    return State(q, qd, x, xd, None)





















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