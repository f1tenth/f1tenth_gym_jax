import chex
from flax import struct


@struct.dataclass
class State:
    """
    Basic Jittable state for cars
    """

    # gym stuff
    rewards: chex.Array  # [n_agent, ]
    done: chex.Array  # [n_agent, ]
    step: int

    # dynamic states
    cartesian_states: (
        chex.Array
    )  # [n_agent, [x, y, delta, v, psi, (psi_dot, beta)]], extra states for st in ()
    frenet_states: chex.Array  # [n_agent, [s, ey, epsi]]
    collisions: chex.Array  # [n_agent,]
    
    # race stuff
    num_laps: chex.Array  # [n_agent, ]
    
    # laser scans TODO: might not need to be part of the state since doesn't depend on previous
    scans: chex.Array  # [n_agent, n_rays]

    # winding vector
    prev_winding_vector: chex.Array  # [n_agent, 2]
    accumulated_angles: chex.Array # [n_agent, 1]

    


@struct.dataclass
class Param:
    """
    Default jittable params for dynamics
    """

    mu: float = 1.0489  # surface friction coefficient
    C_Sf: float = 4.718  # Cornering stiffness coefficient, front
    C_Sr: float = 5.4562  # Cornering stiffness coefficient, rear
    lf: float = 0.15875  # Distance from center of gravity to front axle
    lr: float = 0.17145  # Distance from center of gravity to rear axle
    h: float = 0.074  # Height of center of gravity
    m: float = 3.74  # Total mass of the vehicle
    I: float = 0.04712  # Moment of inertial of the entire vehicle about the z axis
    s_min: float = -0.4189  # Minimum steering angle constraint
    s_max: float = 0.4189  # Maximum steering angle constraint
    sv_min: float = -3.2  # Minimum steering velocity constraint
    sv_max: float = 3.2  # Maximum steering velocity constraint
    v_switch: float = (
        7.319  # Switching velocity (velocity at which the acceleration is no longer able to #spin)
    )
    a_max: float = 9.51  # Maximum longitudinal acceleration
    v_min: float = -5.0  # Minimum longitudinal velocity
    v_max: float = 20.0  # Maximum longitudinal velocity
    width: float = 0.31  # width of the vehicle in meters
    length: float = 0.58  # length of the vehicle in meters
    timestep: float = 0.01  # physical time steps of the dynamics model
    longitudinal_action_type: str = "acceleration"  # speed or acceleration
    steering_action_type: str = (
        "steering_velocity"  # steering_angle or steering_velocity
    )
    integrator: str = "rk4"  # dynamics integrator
    model: str = "st"  # dynamics model type
    produce_scans: bool = False  # whether to turn on laser scan
    theta_dis: int = 2000  # number of discretization in theta, scan param
    fov: float = 4.7  # field of view of the scan, scan param
    num_beams: int = 64  # number of beams in each scan, scan param
    eps: float = 0.01  # epsilon to stop ray marching, scan param
    max_range: float = 10.0  # max range of scan, scan param
    observe_others: bool = True  # whether can observe other agents
    num_rays: float = 1000  # number of rays in each scan
    map_name: str = "Spielberg"  # map for environment
    max_num_laps: int = 1  # maximum number of laps to run before done