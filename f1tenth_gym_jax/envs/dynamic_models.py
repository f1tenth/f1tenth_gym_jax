"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of CommonRoad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng, Renukanandan Tumu
"""

from functools import partial

import chex
import jax

# jax
import jax.numpy as jnp

from .utils import Param


@partial(jax.jit, static_argnums=[1, 2])
def upper_accel_limit(vel, a_max, v_switch):
    """
    Compute the positive longitudinal acceleration limit.

    Parameters
    ----------
    vel : float
        Current vehicle speed.
    a_max : float
        Maximum allowed acceleration magnitude.
    v_switch : float
        Speed above which the acceleration limit decays to avoid wheel slip.

    Returns
    -------
    float
        Positive acceleration limit at the current speed.
    """
    # if vel > v_switch:
    #     pos_limit = a_max * (v_switch / vel)
    # else:
    #     pos_limit = a_max
    pos_limit = jax.lax.select(vel > v_switch, a_max * (v_switch / vel), a_max)
    return pos_limit


@partial(jax.jit, static_argnums=[2, 3, 4, 5])
def accl_constraints(vel, a_long_d, v_switch, a_max, v_min, v_max):
    """
    Apply velocity and acceleration bounds to desired acceleration.

    Parameters
    ----------
    vel : float
        Current vehicle speed.
    a_long_d : float
        Unconstrained desired longitudinal acceleration.
    v_switch : float
        Speed above which the positive acceleration limit decays.
    a_max : float
        Maximum acceleration magnitude.
    v_min : float
        Minimum allowed velocity.
    v_max : float
        Maximum allowed velocity.

    Returns
    -------
    float
        Bounded longitudinal acceleration.
    """

    uac = upper_accel_limit(vel, a_max, v_switch)

    # if (vel <= v_min and a_long_d <= 0) or (vel >= v_max and a_long_d >= 0):
    #     a_long = 0.0
    # elif a_long_d <= -a_max:
    #     a_long = -a_max
    # elif a_long_d >= uac:
    #     a_long = uac
    # else:
    #     a_long = a_long_d

    a_long = jnp.select(
        [
            jnp.logical_or(
                jnp.logical_and(vel <= v_min, a_long_d <= 0),
                jnp.logical_and(vel >= v_max, a_long_d >= 0),
            ),
            (a_long_d <= -a_max),
            (a_long_d >= uac),
        ],
        [0.0, -a_max, uac],
        a_long_d,
    )

    return a_long


@partial(jax.jit, static_argnums=[2, 3, 4, 5])
def steering_constraint(
    steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max
):
    """
    Apply steering angle and steering-rate bounds.

    Parameters
    ----------
    steering_angle : float
        Current front-wheel steering angle.
    steering_velocity : float
        Unconstrained desired steering velocity.
    s_min : float
        Minimum steering angle.
    s_max : float
        Maximum steering angle.
    sv_min : float
        Minimum steering velocity.
    sv_max : float
        Maximum steering velocity.

    Returns
    -------
    float
        Bounded steering velocity.
    """

    # constraint steering velocity
    # if (steering_angle <= s_min and steering_velocity <= 0) or (
    #     steering_angle >= s_max and steering_velocity >= 0
    # ):
    #     steering_velocity = 0.0
    # elif steering_velocity <= sv_min:
    #     steering_velocity = sv_min
    # elif steering_velocity >= sv_max:
    #     steering_velocity = sv_max

    steering_velocity = jnp.select(
        [
            jnp.logical_or(
                jnp.logical_and(steering_angle <= s_min, steering_velocity <= 0),
                jnp.logical_and(steering_angle >= s_max, steering_velocity >= 0),
            ),
            (steering_velocity <= sv_min),
            (steering_velocity >= sv_max),
        ],
        [0.0, sv_min, sv_max],
        steering_velocity,
    )
    return steering_velocity


@partial(jax.jit, static_argnums=[1])
def vehicle_dynamics_ks(x_and_u: chex.Array, params: Param) -> chex.Array:
    """
    Evaluate the kinematic single-track model.

    The implementation follows section 5 of the CommonRoad vehicle models
    reference.

    Parameters
    ----------
    x_and_u : chex.Array, shape (7,)
        State and control vector
        ``[x, y, delta, v, psi, steering_command, longitudinal_command]``.
    params : Param
        Jittable simulation parameters, including geometry, control limits,
        action modes, and timestep.

    Returns
    -------
    chex.Array, shape (7,)
        Right-hand side of the kinematic differential equations with two dummy
        control dimensions appended.
    """
    # Controls
    DELTA = x_and_u[2]
    V = x_and_u[3]
    PSI = x_and_u[4]
    # wheelbase
    lwb = params.lf + params.lr

    # control type
    if params.steering_action_type == "steeringvelocity":
        STEER_VEL = x_and_u[5]
    elif params.steering_action_type == "steeringangle":
        STEER_VEL = (x_and_u[5] - DELTA) / params.timestep

    if params.longitudinal_action_type == "acceleration":
        ACCL = x_and_u[6]
    elif params.longitudinal_action_type == "velocity":
        ACCL = (x_and_u[6] - V) / params.timestep

    # Controls w/ constraints
    STEER_VEL = steering_constraint(
        DELTA, STEER_VEL, params.s_min, params.s_max, params.sv_min, params.sv_max
    )
    ACCL = accl_constraints(
        V, ACCL, params.v_switch, params.a_max, params.v_min, params.v_max
    )

    # system dynamics
    f = jnp.array(
        [
            V * jnp.cos(PSI),  # X_DOT
            V * jnp.sin(PSI),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            (V / lwb) * jnp.tan(DELTA),  # PSI_DOT
            0.0,  # dummy dim
            0.0,  # dummy dim
        ]
    )
    return f


@partial(jax.jit, static_argnums=[1])
def vehicle_dynamics_st_switching(x_and_u: chex.Array, params: Param) -> chex.Array:
    """
    Evaluate the switching dynamic single-track model.

    The implementation follows section 7 of the CommonRoad vehicle models
    reference and switches to a kinematic branch at low speed.

    Parameters
    ----------
    x_and_u : chex.Array, shape (9,)
        State and control vector
        ``[x, y, delta, v, psi, psi_dot, beta, steering_command,
        longitudinal_command]``.
    params : Param
        Jittable simulation parameters, including geometry, tire coefficients,
        control limits, action modes, and timestep.

    Returns
    -------
    chex.Array, shape (9,)
        Right-hand side of the single-track differential equations with two
        dummy control dimensions appended.
    """
    # States
    DELTA = x_and_u[2]
    V = jnp.clip(x_and_u[3], min=0.001)
    PSI = x_and_u[4]
    PSI_DOT = x_and_u[5]
    BETA = x_and_u[6]
    # We have to wrap the slip angle to [-pi, pi]
    # BETA = jnp.arctan2(jnp.sin(BETA), jnp.cos(BETA))

    # gravity constant m/s^2
    g = 9.81

    # control type
    if params.steering_action_type == "steeringvelocity":
        STEER_VEL = x_and_u[7]
    elif params.steering_action_type == "steeringangle":
        STEER_VEL = (x_and_u[7] - DELTA) / params.timestep

    if params.longitudinal_action_type == "acceleration":
        ACCL = x_and_u[8]
    elif params.longitudinal_action_type == "velocity":
        ACCL = (x_and_u[8] - V) / params.timestep

    # Controls w/ constraints
    STEER_VEL = steering_constraint(
        DELTA, STEER_VEL, params.s_min, params.s_max, params.sv_min, params.sv_max
    )
    ACCL = accl_constraints(
        V, ACCL, params.v_switch, params.a_max, params.v_min, params.v_max
    )

    # switch to kinematic model for small velocities
    # wheelbase
    lwb = params.lf + params.lr
    BETA_HAT = jnp.arctan(jnp.tan(DELTA) * params.lr / lwb)
    BETA_DOT = (
        (1 / (1 + (jnp.tan(DELTA) * (params.lr / lwb)) ** 2))
        * (params.lr / (lwb * jnp.cos(DELTA) ** 2))
        * STEER_VEL
    )
    f_ks = jnp.array(
        [
            V * jnp.cos(PSI + BETA_HAT),  # X_DOT
            V * jnp.sin(PSI + BETA_HAT),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            V * jnp.cos(BETA_HAT) * jnp.tan(DELTA) / lwb,  # PSI_DOT
            (1 / lwb)
            * (
                ACCL * jnp.cos(BETA) * jnp.tan(DELTA)
                - V * jnp.sin(BETA) * jnp.tan(DELTA) * BETA_DOT
                + ((V * jnp.cos(BETA) * STEER_VEL) / (jnp.cos(DELTA) ** 2))
            ),  # PSI_DOT_DOT
            BETA_DOT,  # BETA_DOT
            0.0,  # dummy dim
            0.0,  # dummy dim
        ]
    )

    # single track (higher speed) system dynamics
    f = jnp.array(
        [
            V * jnp.cos(PSI + BETA),  # X_DOT
            V * jnp.sin(PSI + BETA),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            PSI_DOT,  # PSI_DOT
            ((params.mu * params.m) / (params.I * (params.lf + params.lr)))
            * (
                params.lf * params.C_Sf * (g * params.lr - ACCL * params.h) * DELTA
                + (
                    params.lr * params.C_Sr * (g * params.lf + ACCL * params.h)
                    - params.lf * params.C_Sf * (g * params.lr - ACCL * params.h)
                )
                * BETA
                - (
                    params.lf
                    * params.lf
                    * params.C_Sf
                    * (g * params.lr - ACCL * params.h)
                    + params.lr
                    * params.lr
                    * params.C_Sr
                    * (g * params.lf + ACCL * params.h)
                )
                * (PSI_DOT / V)
            ),  # PSI_DOT_DOT
            (params.mu / (V * (params.lr + params.lf)))
            * (
                params.C_Sf * (g * params.lr - ACCL * params.h) * DELTA
                - (
                    params.C_Sr * (g * params.lf + ACCL * params.h)
                    + params.C_Sf * (g * params.lr - ACCL * params.h)
                )
                * BETA
                + (
                    params.C_Sr * (g * params.lf + ACCL * params.h) * params.lr
                    - params.C_Sf * (g * params.lr - ACCL * params.h) * params.lf
                )
                * (PSI_DOT / V)
            )
            - PSI_DOT,  # BETA_DOT
            0.0,  # dummy dim
            0.0,  # dummy dim
        ]
    )

    f_ret = jax.lax.select(jnp.abs(V) < 1.5, f_ks, f)

    return f_ret


@partial(jax.jit, static_argnums=[1])
def vehicle_dynamics_st_smooth(x_and_u: chex.Array, params: Param) -> chex.Array:
    """
    Evaluate the smoothly blended dynamic single-track model.

    The implementation follows section 7 of the CommonRoad vehicle models
    reference and blends the dynamic and low-speed kinematic branches.

    Parameters
    ----------
    x_and_u : chex.Array, shape (9,)
        State and control vector
        ``[x, y, delta, v, psi, psi_dot, beta, steering_command,
        longitudinal_command]``.
    params : Param
        Jittable simulation parameters, including geometry, tire coefficients,
        control limits, action modes, and timestep.

    Returns
    -------
    chex.Array, shape (9,)
        Right-hand side of the blended single-track differential equations with
        two dummy control dimensions appended.
    """
    # States
    DELTA = x_and_u[2]
    V = jnp.clip(x_and_u[3], min=0.001)
    PSI = x_and_u[4]
    PSI_DOT = x_and_u[5]
    BETA = x_and_u[6]
    # We have to wrap the slip angle to [-pi, pi]
    # BETA = jnp.arctan2(jnp.sin(BETA), jnp.cos(BETA))

    # gravity constant m/s^2
    g = 9.81

    # control type
    if params.steering_action_type == "steeringvelocity":
        STEER_VEL = x_and_u[7]
    elif params.steering_action_type == "steeringangle":
        STEER_VEL = (x_and_u[7] - DELTA) / params.timestep

    if params.longitudinal_action_type == "acceleration":
        ACCL = x_and_u[8]
    elif params.longitudinal_action_type == "velocity":
        ACCL = (x_and_u[8] - V) / params.timestep

    # Controls w/ constraints
    STEER_VEL = steering_constraint(
        DELTA, STEER_VEL, params.s_min, params.s_max, params.sv_min, params.sv_max
    )
    ACCL = accl_constraints(
        V, ACCL, params.v_switch, params.a_max, params.v_min, params.v_max
    )

    # switch to kinematic model for small velocities
    # wheelbase
    lwb = params.lf + params.lr
    BETA_HAT = jnp.arctan(jnp.tan(DELTA) * params.lr / lwb)
    BETA_DOT = (
        (1 / (1 + (jnp.tan(DELTA) * (params.lr / lwb)) ** 2))
        * (params.lr / (lwb * jnp.cos(DELTA) ** 2))
        * STEER_VEL
    )
    f_ks = jnp.array(
        [
            V * jnp.cos(PSI + BETA_HAT),  # X_DOT
            V * jnp.sin(PSI + BETA_HAT),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            V * jnp.cos(BETA_HAT) * jnp.tan(DELTA) / lwb,  # PSI_DOT
            (1 / lwb)
            * (
                ACCL * jnp.cos(BETA) * jnp.tan(DELTA)
                - V * jnp.sin(BETA) * jnp.tan(DELTA) * BETA_DOT
                + ((V * jnp.cos(BETA) * STEER_VEL) / (jnp.cos(DELTA) ** 2))
            ),  # PSI_DOT_DOT
            BETA_DOT,  # BETA_DOT
            0.0,  # dummy dim
            0.0,  # dummy dim
        ]
    )

    # single track (higher speed) system dynamics
    f = jnp.array(
        [
            V * jnp.cos(PSI + BETA),  # X_DOT
            V * jnp.sin(PSI + BETA),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            PSI_DOT,  # PSI_DOT
            ((params.mu * params.m) / (params.I * (params.lf + params.lr)))
            * (
                params.lf * params.C_Sf * (g * params.lr - ACCL * params.h) * DELTA
                + (
                    params.lr * params.C_Sr * (g * params.lf + ACCL * params.h)
                    - params.lf * params.C_Sf * (g * params.lr - ACCL * params.h)
                )
                * BETA
                - (
                    params.lf
                    * params.lf
                    * params.C_Sf
                    * (g * params.lr - ACCL * params.h)
                    + params.lr
                    * params.lr
                    * params.C_Sr
                    * (g * params.lf + ACCL * params.h)
                )
                * (PSI_DOT / V)
            ),  # PSI_DOT_DOT
            (params.mu / (V * (params.lr + params.lf)))
            * (
                params.C_Sf * (g * params.lr - ACCL * params.h) * DELTA
                - (
                    params.C_Sr * (g * params.lf + ACCL * params.h)
                    + params.C_Sf * (g * params.lr - ACCL * params.h)
                )
                * BETA
                + (
                    params.C_Sr * (g * params.lf + ACCL * params.h) * params.lr
                    - params.C_Sf * (g * params.lr - ACCL * params.h) * params.lf
                )
                * (PSI_DOT / V)
            )
            - PSI_DOT,  # BETA_DOT
            0.0,  # dummy dim
            0.0,  # dummy dim
        ]
    )

    weight_ks = sigmoid_interp(jnp.abs(V))
    f_interp = f_ks * weight_ks + f * (1.0 - weight_ks)

    return f_interp


@jax.jit
def sigmoid_interp(x, shift=0.55, scale=100.0):
    weight = jax.nn.sigmoid(scale * (shift - x))
    return jnp.round(weight, 3)


@partial(jax.jit, static_argnums=[2])
def pid_steer(steer, current_steer, max_sv):
    # steering
    steer_diff = steer - current_steer
    # if np.fabs(steer_diff) > 1e-4:
    #     sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    # else:
    #     sv = 0.0
    sv = jax.lax.select(
        jnp.fabs(steer_diff) > 1e-4,
        (steer_diff / jnp.fabs(steer_diff)) * max_sv,
        0.0,
    )

    return sv


@jax.jit
def pid_accl(speed, current_speed, max_a, max_v, min_v):
    """
    Convert desired speed to longitudinal acceleration.

    Parameters
    ----------
    speed : float
        Desired vehicle speed.
    current_speed : float
        Current vehicle speed.
    max_a : float
        Maximum acceleration magnitude.
    max_v : float
        Maximum allowed velocity.
    min_v : float
        Minimum allowed velocity.

    Returns
    -------
    float
        Desired longitudinal acceleration.
    """
    # accl
    vel_diff = speed - current_speed

    # currently forward
    # if current_speed > 0.0:
    #     if vel_diff > 0:
    #         # accelerate
    #         accl = (10.0 * max_a / max_v) * vel_diff
    #     else:
    #         # braking
    #         accl = (10.0 * max_a / (-min_v)) * vel_diff
    # # currently backwards
    # else:
    #     if vel_diff > 0:
    #         # braking
    #         accl = (2.0 * max_a / max_v) * vel_diff
    #     else:
    #         # accelerating
    #         accl = (2.0 * max_a / (-min_v)) * vel_diff

    accl = jnp.select(
        [
            jnp.logical_and(current_speed > 0.0, vel_diff > 0.0),
            jnp.logical_and(current_speed > 0.0, vel_diff <= 0.0),
            jnp.logical_and(current_speed <= 0.0, vel_diff > 0.0),
            jnp.logical_and(current_speed <= 0.0, vel_diff <= 0.0),
        ],
        [
            (10.0 * max_a / max_v) * vel_diff,
            (10.0 * max_a / (-min_v)) * vel_diff,
            (2.0 * max_a / max_v) * vel_diff,
            (2.0 * max_a / (-min_v)) * vel_diff,
        ],
        0.0,
    )

    return accl
