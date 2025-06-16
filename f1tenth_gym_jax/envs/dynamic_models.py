"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of commanroad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng, Renukanandan Tumu
"""

# jax
import jax.numpy as jnp
import jax
import chex

# others
import numpy as np
from functools import partial

from .utils import Param


@partial(jax.jit, static_argnums=[1, 2])
def upper_accel_limit(vel, a_max, v_switch):
    """
    Upper acceleration limit, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            a_max (float): maximum allowed acceleration, symmetrical
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)

        Returns:
            positive_accel_limit (float): adjusted acceleration
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
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            a_long_d (float): unconstrained desired acceleration in the direction of travel.
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration, symmetrical
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
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
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
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
    Single Track Kinematic Vehicle Dynamics.
    Follows https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 5

        Args:
            x_and_u (jax.numpy.ndarray (7, )): vehicle state vector with control input vector (x0, x1, x2, x3, x4, u0, u1)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
                u0: steering angle velocity of front wheels
                u1: longitudinal acceleration
            params (Param): jittable dataclass with the following fields:
                mu (float): friction coefficient
                C_Sf (float): cornering stiffness of front wheels
                C_Sr (float): cornering stiffness of rear wheels
                lf (float): distance from center of gravity to front axle
                lr (float): distance from center of gravity to rear axle
                h (float): height of center of gravity
                m (float): mass of vehicle
                I (float): moment of inertia of vehicle, about Z axis
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity
                v_switch (float): velocity above which the acceleration is no longer able to create wheel slip
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

        Returns:
            f (jax.numpy.ndarray (7, )): right hand side of differential equations
    """
    # Controls
    X = x_and_u[0]
    Y = x_and_u[1]
    DELTA = x_and_u[2]
    V = x_and_u[3]
    PSI = x_and_u[4]
    # wheelbase
    lwb = params.lf + params.lr

    # constrained controls
    STEER_VEL = steering_constraint(
        DELTA, x_and_u[5], params.s_min, params.s_max, params.sv_min, params.sv_max
    )
    ACCL = accl_constraints(
        V, x_and_u[6], params.v_switch, params.a_max, params.v_min, params.v_max
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
    Single Track Vehicle Dynamics.
    From https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 7

        Args:
            x_and_u (jax.numpy.ndarray (7, )): vehicle state vector with control input vector (x0, x1, x2, x3, x4, u0, u1)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
                u0: steering angle velocity of front wheels
                u1: longitudinal acceleration
            params (Param): jittable dataclass with the following fields:
                mu (float): friction coefficient
                C_Sf (float): cornering stiffness of front wheels
                C_Sr (float): cornering stiffness of rear wheels
                lf (float): distance from center of gravity to front axle
                lr (float): distance from center of gravity to rear axle
                h (float): height of center of gravity
                m (float): mass of vehicle
                I (float): moment of inertia of vehicle, about Z axis
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity
                v_switch (float): velocity above which the acceleration is no longer able to create wheel slip
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

        Returns:
            f (jax.numpy.ndarray (7, )): right hand side of differential equations
    """
    # States
    X = x_and_u[0]
    Y = x_and_u[1]
    DELTA = x_and_u[2]
    V = jnp.clip(x_and_u[3], min=0.001)
    PSI = x_and_u[4]
    PSI_DOT = x_and_u[5]
    BETA = x_and_u[6]
    # We have to wrap the slip angle to [-pi, pi]
    # BETA = jnp.arctan2(jnp.sin(BETA), jnp.cos(BETA))

    # gravity constant m/s^2
    g = 9.81

    # Controls w/ constraints
    STEER_VEL = steering_constraint(
        DELTA, x_and_u[7], params.s_min, params.s_max, params.sv_min, params.sv_max
    )
    ACCL = accl_constraints(
        V, x_and_u[8], params.v_switch, params.a_max, params.v_min, params.v_max
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
    Single Track Vehicle Dynamics.
    From https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 7

        Args:
            x_and_u (jax.numpy.ndarray (7, )): vehicle state vector with control input vector (x0, x1, x2, x3, x4, u0, u1)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
                u0: steering angle velocity of front wheels
                u1: longitudinal acceleration
            params (Param): jittable dataclass with the following fields:
                mu (float): friction coefficient
                C_Sf (float): cornering stiffness of front wheels
                C_Sr (float): cornering stiffness of rear wheels
                lf (float): distance from center of gravity to front axle
                lr (float): distance from center of gravity to rear axle
                h (float): height of center of gravity
                m (float): mass of vehicle
                I (float): moment of inertia of vehicle, about Z axis
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity
                v_switch (float): velocity above which the acceleration is no longer able to create wheel slip
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

        Returns:
            f (jax.numpy.ndarray (7, )): right hand side of differential equations
    """
    # States
    X = x_and_u[0]
    Y = x_and_u[1]
    DELTA = x_and_u[2]
    V = x_and_u[3]
    PSI = x_and_u[4]
    PSI_DOT = x_and_u[5]
    BETA = x_and_u[6]
    # We have to wrap the slip angle to [-pi, pi]
    # BETA = jnp.arctan2(jnp.sin(BETA), jnp.cos(BETA))

    # gravity constant m/s^2
    g = 9.81

    # Controls w/ constraints
    STEER_VEL = steering_constraint(
        DELTA, x_and_u[7], params.s_min, params.s_max, params.sv_min, params.sv_max
    )
    ACCL = accl_constraints(
        V, x_and_u[8], params.v_switch, params.a_max, params.v_min, params.v_max
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
    weight = jnp.exp(scale * (x - shift)) / (1.0 + jnp.exp(scale * (x - shift)))
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
        jnp.fabs(steer_diff) > 1e-4, (steer_diff / np.fabs(steer_diff)) * max_sv, 0.0
    )

    return sv


@jax.jit
def pid_accl(speed, current_speed, max_a, max_v, min_v):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
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
            (current_speed > 0.0 and vel_diff > 0.0),
            (current_speed > 0.0 and vel_diff <= 0.0),
            (current_speed <= 0.0 and vel_diff > 0.0),
            (current_speed <= 0.0 and vel_diff <= 0.0),
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
