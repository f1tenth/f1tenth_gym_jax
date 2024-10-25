"""
Prototype of Utility functions and GJK algorithm / Separating Axis Theorem for Collision checks between vehicles
Originally from https://github.com/kroitor/gjk.c
Author: Hongrui Zheng
"""

import numpy as np
import jax
import jax.numpy as jnp
import chex

import os
from functools import partial


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
    edges = jnp.roll(vertices, 1, axis=1) - vertices
    center_p = pixel_centers[:, None, None] - edges
    cross_prods = jnp.cross(center_p, edges)
    left_of = jnp.where(cross_prods <= 0, 1.0, 0.0)
    all_left_of = jnp.sum(left_of, axis=-1)
    collisions = jnp.where(
        jnp.sum(jnp.where(all_left_of == 4.0, 1.0, 0.0), axis=0) > 0.0, 1.0, 0.0
    )
    return collisions


@jax.jit
def perpendicular(pt):
    """
    Return a 2-vector's perpendicular vector

    Args:
        pt (np.ndarray, (2,)): input vector

    Returns:
        pt (np.ndarray, (2,)): perpendicular vector
    """
    pt = jnp.flip(pt)
    pt = pt.at[1].multiply(-1)
    return pt


@jax.jit
def tripleProduct(a, b, c):
    """
    Return triple product of three vectors

    Args:
        a, b, c (jax.numpy.ndarray, (2,)): input vectors

    Returns:
        (jax.numpy.ndarray, (2,)): triple product
    """
    ac = jnp.dot(a, c)
    bc = jnp.dot(b, c)
    return b * ac - a * bc


@jax.jit
def avgPoint(vertices):
    """
    Return the average point of multiple vertices

    Args:
        vertices (jax.numpy.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        avg (jax.numpy.ndarray, (2,)): average point of the vertices
    """
    return jnp.sum(vertices, axis=0) / vertices.shape[0]


@jax.jit
def indexOfFurthestPoint(vertices, d):
    """
    Return the index of the vertex furthest away along a direction in the list of vertices

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        idx (int): index of the furthest point
    """
    return jnp.argmax(jnp.dot(vertices, d))


@jax.jit
def support(vertices1, vertices2, d):
    """
    Minkowski sum support function for GJK

    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body
        d (np.ndarray, (2, )): direction to find the support along

    Returns:
        support (np.ndarray, (n, 2)): Minkowski sum
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@jax.jit
def collision_gjk(vertices1, vertices2):
    """
    GJK test to see whether two bodies overlap

    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body

    Returns:
        overlap (boolean): True if two bodies collide
    """
    index = 0
    simplex = jnp.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    d.at[0].set(jax.lax.select((d[0] == 0 and d[1] == 0), 1.0, d[0]))

    a = support(vertices1, vertices2, d)
    simplex.at[index, :].set(a)

    if d.dot(a) <= 0:
        return False

    @jax.jit
    def collision_loop():
        d = -a
        iter_count = 0

        def cond_fn(state):
            iter_count, index, simplex, d, a, bool_ret = state
            return iter_count < 1e3

        def body_fn(state):
            iter_count, index, simplex, d, a, bool_ret = state
            a = support(vertices1, vertices2, d)
            index += 1
            simplex = simplex.at[index, :].set(a)
            bool_ret = d.dot(a) > 0
            state = (iter_count, index, simplex, d, a, bool_ret)

            ao = -a

            def branch_fn(state):
                iter_count, index, simplex, d, a, bool_ret = state
                b = simplex[0, :]
                ab = b - a
                d = tripleProduct(ab, ao, ab)
                d = jax.lax.cond(
                    jnp.linalg.norm(d) < 1e-10, lambda _: perpendicular(ab), lambda _: d
                )
                return iter_count, index, simplex, d, a, bool_ret

            def branch_fn2(state):
                iter_count, index, simplex, d, a, bool_ret = state
                b = simplex[1, :]
                c = simplex[0, :]
                ab = b - a
                ac = c - a
                acperp = tripleProduct(ab, ac, ac)
                # TODO: double check this part
                d = jax.lax.cond(
                    acperp.dot(ao) >= 0,
                    lambda _: acperp,
                    lambda _: tripleProduct(ac, ab, ab),
                )
                bool_ret = jnp.dot(d, ao) < 0 and bool_ret
                simplex = jax.lax.cond(
                    acperp.dot(ao) < 0,
                    lambda _: simplex.at[0, :].set(simplex[1, :]),
                    lambda _: simplex,
                )
                simplex = simplex.at[1, :].set(simplex[2, :])
                index -= 1
                return iter_count, index, simplex, d, a, bool_ret

            state = jax.lax.cond(index < 2, branch_fn, branch_fn2, state)
            iter_count += 1
            return iter_count, index, simplex, d, a

        state = (0, index, simplex, d, a, False)
        state = jax.lax.while_loop(cond_fn, body_fn, state)
        return state[4].dot(state[3]) <= 0

        while iter_count < 1e3:
            a = support(vertices1, vertices2, d)
            index += 1
            simplex[index, :] = a
            if d.dot(a) <= 0:
                return False

            ao = -a

            if index < 2:
                b = simplex[0, :]
                ab = b - a
                d = tripleProduct(ab, ao, ab)
                if np.linalg.norm(d) < 1e-10:
                    d = perpendicular(ab)
                continue

            b = simplex[1, :]
            c = simplex[0, :]
            ab = b - a
            ac = c - a

            acperp = tripleProduct(ab, ac, ac)

            if acperp.dot(ao) >= 0:
                d = acperp
            else:
                abperp = tripleProduct(ac, ab, ab)
                if abperp.dot(ao) < 0:
                    return True
                simplex[0, :] = simplex[1, :]
                d = abperp

            simplex[1, :] = simplex[2, :]
            index -= 1

            iter_count += 1
        return False
        return in_collision

    ret = jax.lax.cond(jnp.dot(d, a) <= 0, collision_loop, lambda x: False)

    return ret


@njit(cache=True)
def collision_multiple(vertices):
    """
    Check pair-wise collisions for all provided vertices

    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): all vertices for checking pair-wise collision

    Returns:
        collisions (np.ndarray (num_vertices, )): whether each body is in collision
        collision_idx (np.ndarray (num_vertices, )): which index of other body is each index's body is in collision, -1 if not in collision
    """
    collisions = np.zeros((vertices.shape[0],))
    collision_idx = -1 * np.ones((vertices.shape[0],))
    # looping over all pairs
    for i in range(vertices.shape[0] - 1):
        for j in range(i + 1, vertices.shape[0]):
            # check collision
            vi = np.ascontiguousarray(vertices[i, :, :])
            vj = np.ascontiguousarray(vertices[j, :, :])
            ij_collision = collision(vi, vj)
            # fill in results
            if ij_collision:
                collisions[i] = 1.0
                collisions[j] = 1.0
                collision_idx[i] = j
                collision_idx[j] = i

    return collisions, collision_idx


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
    rl = H.dot(np.asarray([[-length / 2], [width / 2], [0.0], [1.0]])).flatten()
    rr = H.dot(np.asarray([[-length / 2], [-width / 2], [0.0], [1.0]])).flatten()
    fl = H.dot(np.asarray([[length / 2], [width / 2], [0.0], [1.0]])).flatten()
    fr = H.dot(np.asarray([[length / 2], [-width / 2], [0.0], [1.0]])).flatten()
    rl = rl / rl[3]
    rr = rr / rr[3]
    fl = fl / fl[3]
    fr = fr / fr[3]
    vertices = np.asarray(
        [[rl[0], rl[1]], [rr[0], rr[1]], [fr[0], fr[1]], [fl[0], fl[1]]]
    )
    return vertices
