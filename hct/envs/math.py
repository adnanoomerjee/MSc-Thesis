from brax.math import *
from brax.base import Transform, Motion

import jax.numpy as jp
from typing import Optional

import types, functools
from multipledispatch import dispatch

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

def quaternion_to_spherical(quat: jp.ndarray) -> jp.ndarray:
    w = quat[0]
    xyz = quat[1:]
    # Convert quaternion to cartesian axis-angle
    v = normalize(xyz)
    angle = 2*safe_arccos(w) - jp.pi
    # convert cartesian axis-angle to spherical
    r = angle
    theta = safe_arccos(v[3])
    phi = jp.arctan2(v[2], v[1])
    return jp.array(r, theta, phi)

def world_to_egocentric(t) -> Transform:
    """Converts a transform in world frame to transform in egocentric frame.

    Args:
        t: transform in world frame

    Returns:
        t': transform in egocentric frame
    """
    r = t.take(0)
    return t.vmap(in_axes=(0,None)).to_local(r)

def egocentric_to_world(t) -> Transform:
    """Converts a transform in egocentric frame to transform in world frame.

    Args:
        t: transform in egocentric frame

    Returns:
        t': transform in world frame
    """
    r = t.take(0)
    return r.vmap(in_axes=(None,0)).do(t)

@jax.vmap
def dist_quat(quat1, quat2):
    """Accounts for double cover property of quaternions"""
    v1 = quat1 - quat2
    v2 = quat1 + quat2
    innerprod1 = jp.dot(v1,v1)
    innerprod2 = jp.dot(v2,v2)
    return jax.lax.select(innerprod1<innerprod2, v1, v2)

def random_ordered_subset(rng, idx: jp.ndarray):
    rng, rng1, rng2 = jax.random.split(rng, 3)
    n = jax.random.choice(rng2, jp.array(range(0, len(idx)+1)))
    subset = jax.random.choice(rng1, idx, shape=(n,), replace = False)
    return jp.sort(subset)

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