from __future__ import annotations

import jax

from ..collision_models import get_trmtx


@jax.jit
def _get_tire_vertices(pose_arr, length, width, tire_width, tire_length, fl, steering):
    """
    Utility function to return vertices of the car's tire given pose and size

    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width

    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    working_width = jax.lax.select(fl, width, -width)
    working_tire_width = jax.lax.select(fl, tire_width, -tire_width)

    H_shift = get_trmtx(
        jax.numpy.array(
            [
                -(length / 2 - tire_length / 2),
                -(working_width / 2 - working_tire_width / 2),
                0,
            ]
        )
    )
    H_steer = get_trmtx(jax.numpy.array([0, 0, steering]))
    H_back = get_trmtx(
        jax.numpy.array(
            [
                length / 2 - tire_length / 2,
                working_width / 2 - working_tire_width / 2,
                0,
            ]
        )
    )
    H = get_trmtx(pose_arr)
    H = H.dot(H_back).dot(H_steer).dot(H_shift)
    fl = H.dot(
        jax.numpy.asarray([[length / 2], [working_width / 2], [0.0], [1.0]])
    ).flatten()
    fr = H.dot(
        jax.numpy.asarray(
            [[length / 2], [working_width / 2 - working_tire_width], [0.0], [1.0]]
        )
    ).flatten()
    rr = H.dot(
        jax.numpy.asarray(
            [
                [length / 2 - tire_length],
                [working_width / 2 - working_tire_width],
                [0.0],
                [1.0],
            ]
        )
    ).flatten()
    rl = H.dot(
        jax.numpy.asarray(
            [[length / 2 - tire_length], [working_width / 2], [0.0], [1.0]]
        )
    ).flatten()
    rl = rl / rl[3]
    rr = rr / rr[3]
    fl = fl / fl[3]
    fr = fr / fr[3]
    vertices = jax.numpy.asarray(
        [
            [rl[0], rl[1]],
            [fl[0], fl[1]],
            [fr[0], fr[1]],
            [rr[0], rr[1]],
            [rl[0], rl[1]],
        ]
    )

    return vertices
