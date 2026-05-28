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
    See if two bodies' projections overlap along a normal axis

    Args:
        vertices1 (jax.numpy.ndarray, (n, 2)): vertices of the first body
        vertices2 (jax.numpy.ndarray, (n, 2)): vertices of the second body

    Returns:
        overlap (boolean): True if two projections overlap
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
    SAT test to see whether two bodies overlap

    Args:
        vertices1 (jax.numpy.ndarray, (n, 2)): vertices of the first body
        vertices2 (jax.numpy.ndarray, (n, 2)): vertices of the second body

    Returns:
        overlap (boolean): True if two bodies collide
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
    Check vertices collision with map occupancy
    Rasters car polygon to map occupancy
    vmap across number of cars, and number of occupied pixels
    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): agent rectangle vertices, ccw winding order
        pixel_centers (np.ndarray (HxW, 2)): x, y position of pixel centers of map image
    Returns:
        collisions (np.ndarray (num_bodies, )): whether each body is in collision with map
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
    Get transformation matrix of vehicle frame -> global frame

    Args:
        pose (np.ndarray (3, )): current pose of the vehicle

    return:
        H (np.ndarray (4, 4)): transformation matrix
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
    Utility function to return vertices of the car body given pose and size

    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width

    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
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
