import numpy as np
from numba import njit

from .utils import steering_constraint, accl_constraints

@njit(cache=True)
def vehicle_dynamics_ks(
    x,
    u_init,
    mu,
    C_Sf,
    C_Sr,
    lf,
    lr,
    h,
    m,
    I,
    s_min,
    s_max,
    sv_min,
    sv_max,
    v_switch,
    a_max,
    v_min,
    v_max,
):
    """
    Single Track Kinematic Vehicle Dynamics.
    Follows https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 5

        Args:
            x (numpy.ndarray (5, )): vehicle state vector (x0, x1, x2, x3, x4)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration
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
            f (numpy.ndarray): right hand side of differential equations
    """
    # Controls
    X = x[0]
    Y = x[1]
    DELTA = x[2]
    V = x[3]
    PSI = x[4]
    # Raw Actions
    RAW_STEER_VEL = u_init[0]
    RAW_ACCL = u_init[1]
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array(
        [
            steering_constraint(DELTA, RAW_STEER_VEL, s_min, s_max, sv_min, sv_max),
            accl_constraints(V, RAW_ACCL, v_switch, a_max, v_min, v_max),
        ]
    )
    # Corrected Actions
    STEER_VEL = u[0]
    ACCL = u[1]

    # system dynamics
    f = np.array(
        [
            V * np.cos(PSI),  # X_DOT
            V * np.sin(PSI),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            (V / lwb) * np.tan(DELTA),  # PSI_DOT
        ]
    )
    return f