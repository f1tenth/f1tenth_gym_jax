"""
Prototype of Utility functions and GJK algorithm / Separating Axis Theorem for Collision checks between vehicles
Originally from https://github.com/kroitor/gjk.c
Author: Hongrui Zheng
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp


@jax.jit
def sa(normal: chex.Array, vertices1: chex.Array, vertices2: chex.Array) -> bool:
    """
    Check whether two body projections are separated along a normal axis.

    Parameters
    ----------
    normal : chex.Array
        Axis normal used for the projection.
    vertices1 : chex.Array, shape (n, 2)
        Vertices of the first body.
    vertices2 : chex.Array, shape (n, 2)
        Vertices of the second body.

    Returns
    -------
    bool
        True when this axis separates the two projected bodies.
    """

    # project vertices of both bodies onto the axis
    proj1 = jnp.dot(vertices1, normal)
    proj2 = jnp.dot(vertices2, normal)

    # Check if there is an overlap on this axis
    return jnp.logical_not(
        (jnp.max(proj1) >= jnp.min(proj2)) & (jnp.max(proj2) >= jnp.min(proj1))
    )


@jax.jit
def collision(vertices: chex.Array) -> bool:
    """
    Check whether two rectangular bodies overlap with SAT.

    Parameters
    ----------
    vertices : chex.Array, shape (4, 4)
        Concatenated vertices for two rectangular bodies. Columns ``0:2`` are
        the first body and columns ``2:4`` are the second body.

    Returns
    -------
    bool
        True if the bodies collide.
    """
    vertices1 = vertices[:, :2]
    vertices2 = vertices[:, 2:]
    # Find the normals for both rectangles
    vec1 = jnp.roll(vertices1, -1, axis=0) - vertices1
    vec2 = jnp.roll(vertices2, -1, axis=0) - vertices2
    normals = jnp.concatenate(
        (
            jnp.column_stack((-vec1[:, 1], vec1[:, 0])),
            jnp.column_stack((-vec2[:, 1], vec2[:, 0])),
        ),
        axis=0,
    )

    separating_axis = jax.vmap(partial(sa, vertices1=vertices1, vertices2=vertices2))(
        normals
    )

    return jnp.logical_not(jnp.any(separating_axis))


@jax.jit
def collision_map(vertices, pixel_centers):
    """
    Check vehicle polygons against occupied map pixels.

    Parameters
    ----------
    vertices : chex.Array, shape (num_bodies, 4, 2)
        Agent rectangle vertices in counter-clockwise winding order.
    pixel_centers : chex.Array, shape (num_pixels, 2)
        ``x`` and ``y`` positions of occupied map pixel centers.

    Returns
    -------
    chex.Array, shape (num_bodies,)
        Boolean collision flag for each body.
    """
    edges = jnp.roll(vertices, -1, axis=1) - vertices
    point_vecs = pixel_centers[:, None, None, :] - vertices[None, :, :, :]
    cross_prods = jnp.cross(edges[None, :, :, :], point_vecs, axis=-1)
    inside_each = (cross_prods >= 0.0).astype(jnp.float32)
    num_inside = jnp.sum(inside_each, axis=-1)
    collisions = jnp.any(num_inside == 4, axis=0)
    return collisions


"""
Utility functions for getting vertices by pose and shape
"""


@jax.jit
def get_trmtx(pose):
    """
    Compute the transform from vehicle frame to global frame.

    Parameters
    ----------
    pose : chex.Array, shape (3,)
        Vehicle pose ``[x, y, yaw]`` in world coordinates.

    Returns
    -------
    chex.Array, shape (4, 4)
        Homogeneous transformation matrix.
    """
    x = pose[0]
    y = pose[1]
    th = pose[2]
    cos = jnp.cos(th)
    sin = jnp.sin(th)
    H = jnp.array(
        [
            [cos, -sin, 0.0, x],
            [sin, cos, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return H


@jax.jit
def get_vertices(pose, length, width):
    """
    Compute vehicle rectangle vertices from pose and body size.

    Parameters
    ----------
    pose : chex.Array, shape (3,)
        Vehicle pose ``[x, y, yaw]`` in world coordinates.
    length : float
        Vehicle body length.
    width : float
        Vehicle body width.

    Returns
    -------
    chex.Array, shape (4, 2)
        Corner vertices of the vehicle body.
    """
    H = get_trmtx(pose)
    rl = H.dot(jnp.asarray([[-length / 2], [width / 2], [0.0], [1.0]])).flatten()
    rr = H.dot(jnp.asarray([[-length / 2], [-width / 2], [0.0], [1.0]])).flatten()
    fl = H.dot(jnp.asarray([[length / 2], [width / 2], [0.0], [1.0]])).flatten()
    fr = H.dot(jnp.asarray([[length / 2], [-width / 2], [0.0], [1.0]])).flatten()
    rl = rl / rl[3]
    rr = rr / rr[3]
    fl = fl / fl[3]
    fr = fr / fr[3]
    vertices = jnp.asarray(
        [[rl[0], rl[1]], [rr[0], rr[1]], [fr[0], fr[1]], [fl[0], fl[1]]]
    )
    return vertices
