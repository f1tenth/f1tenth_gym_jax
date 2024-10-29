"""
Jax jittable f1tenth_gym environment

Based on JaxMARL api which follows PettingZoo/Gymnax

Author: Hongrui Zheng
"""

# typing
from typing import Any, Dict, Optional, Tuple, Union

from .multi_agent_env import MultiAgentEnv
from .spaces import Box
from .spaces import Tuple as JaxMARLTuple
from .spaces import Dict as JaxMARLDict

# jax
import jax
import jax.numpy as jnp
import chex
from flax import struct

# numpy scipy
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

from .track import Track

# base classes
from .base_classes import Simulator, DynamicModel
from .observation import observation_factory, Observation
from .reset import make_reset_fn

# other
from functools import partial

# dynamics
from .dynamic_models import vehicle_dynamics_ks, vehicle_dynamics_st

# integrators
from .integrator import integrate_euler, integrate_rk4

# scanning
from jax_pf.ray_marching import get_scan

# collisions
from .collision_models import collision, collision_map, get_vertices


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
    collisions: chex.Array  # [n_agent, n_agent + 1]

    # laser scans TODO: might not need to be part of the state since doesn't depend on previous
    scans: chex.Array = None  # [n_agent, n_rays]

    # race stuff
    num_laps: chex.Array  # [n_agent, ]


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


@jax.jit
def ret_orig(x):
    return x


class F110Env(MultiAgentEnv):
    """
    JAX compatible gym environment for F1TENTH

    Args:
        kwargs:
            num_agents (int, default=2): number of agents in the environment
            map (str, default='vegas'): name of the map used for the environment.
            params (Parm): vehicle parameters.

    """

    def __init__(self, num_agents: int = 1, params: Param = Param(), **kwargs):
        super().__init__()
        self.params = params
        # agents
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

        # choose dynamics model and integrators
        if params.integrator == "rk4":
            self.integrator_func = integrate_rk4
        elif params.integrator == "euler":
            self.integrator_func = integrate_euler
        else:
            raise (
                ValueError(
                    f"Chosen integrator {params.integrator} is invalid. Choose either 'rk4' or 'euler'."
                )
            )

        if params.model == "st":
            self.model_func = vehicle_dynamics_st
            self.state_size = 7
        elif params.model == "ks":
            self.model_func = vehicle_dynamics_ks
            self.state_size = 5
        else:
            raise (
                ValueError(
                    f"Chosen dynamics model {params.model} is invalid. Choose either 'st' or 'ks'."
                )
            )

        # spaces
        self.action_spaces = {i: Box(-jnp.inf, jnp.inf, (2,)) for i in self.agents}

        # scanning or not
        if params.produce_scans:
            self.scan_size = params.num_rays
        else:
            self.scan_size = 0

        # observing others
        if params.observe_others:
            # (relative_x, relative_y, relative_psi, longitudinal_v)
            self.all_other_state_size = 4 * (self.num_agents - 1)
        else:
            self.all_other_state_size = 0

        self.observation_spaces = {
            i: Box(
                -jnp.inf,
                jnp.inf,
                (self.state_size + self.all_others_state_size + self.scan_size,),
            )
            for i in self.agents
        }
        self.observation_space_ind = {
            "dynamics_state": list(range(self.state_size)),
            "other_agent_dynamics_state": list(
                range(self.state_size, self.state_size + self.all_other_state_size)
            ),
            "scan": list(
                range(
                    self.state_size,
                    self.state_size + self.all_other_state_size,
                    self.state_size,
                    self.state_size + self.all_other_state_size + self.scan_size,
                )
            ),
        }

        # load map
        self.track = Track.from_track_name(params.map_name)

        # get a interior point of track as winding number looking point
        start_point_curvature = self.track.calc_curvature_jax(0.0)
        self.winding_point = self.track.frenet_to_cartesian_jax(
            s=0.0, ey=jnp.sign(start_point_curvature) * 1.5
        )

        # set pixel centers of occupancy map
        self._set_pixelcenters()

        # TODO: keep all start line, lap information in frenet frame

        # reset modes
        self.reset_fn = make_reset_fn(
            **self.config["reset_config"], track=self.track, num_agents=self.num_agents
        )

        # scan params if produce scan
        if self.params.produce_scans:
            self.fov = self.params.fov
            self.num_beams = self.params.num_beams
            self.theta_dis = self.params.theta_dis
            self.eps = self.params.eps
            self.max_range = self.params.max_range

            angle_increment = self.fov / (self.num_beams - 1)
            self.theta_index_increment = self.theta_dis * angle_increment / (2 * np.pi)
            theta_arr = jnp.linspace(0.0, 2 * jnp.pi, num=self.theta_dis)
            self.scan_sines = jnp.sin(theta_arr)
            self.scan_cosines = jnp.cos(theta_arr)

            self.distance_transform = edt(self.track.occ_map) * self.track.resolution
            self.height, self.width = self.track.occ_map.shape
            self.resolution = self.track.resolution
            self.orig_x = self.track.ox
            self.orig_y = self.track.oy
            self.orig_c = jnp.cos(self.track.oyaw)
            self.orig_s = jnp.sin(self.track.oyaw)

    def _set_pixelcenters(self):
        map_img = self.track.occ_map
        h, w = map_img.shape
        reso = self.track.resolution
        ox = self.track.ox
        oy = self.track.oy
        x_ind, y_ind = np.meshgrid(range(w), range(h))
        pcx = (x_ind * reso + ox + reso / 2).flatten()
        pcy = (y_ind * reso + oy + reso / 2).flatten()
        self.pixel_centers = np.vstack((pcx, pcy)).T
        map_mask = (map_img == 0.0).flatten()
        self.pixel_centers = self.pixel_centers[map_mask, :]

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        # 1. state + scan
        # make x_and_u
        x = state.cartesian_states
        us = jnp.array([actions[i] for i in self.agents])
        x_and_u = jnp.hstack((x, us))
        # integrate dynamics, vmapped
        integrator = jax.vmap(self.integrator_func, in_axes=[0])
        new_x_and_u = integrator(self.model_func, x_and_u, self.params)
        state.cartesian_states = new_x_and_u[:, :-2]
        state = jax.lax.cond(
            self.params.produce_scans, self._scan(state), ret_orig(state)
        )

        # 2. collisions
        state = self._collisions(state)

        # 2. get obs
        obs = self.get_obs(state)

        # 3. dones
        dones = self.check_dones(state)

        # 4. rewards
        rewards = self.get_rewards(state)

        # 5. info
        infos = {}

        return obs, state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        # TODO: reset lap counters etc
        self.num_laps = jnp.zeros((self.num_agents,), dtype=int)
        self.accumulated_angles = jnp.zeros(
            self.num_agents,
        )
        # TODO: reset states
        # TODO: get obs

        state = State(
            rewards=jnp.zeros((self.num_agents,)),
            done=jnp.full((self.num_agents), False),
            step=0,
            cartesian_states=jnp.zeros((self.num_agents, self.state_size)),
            frenet_states=jnp.zeros((self.num_agents, self.frenet_state_size)),
            num_laps=jnp.full((self.num_agents), 0),
        )
        state = jax.lax.cond(
            self.params.produce_scans, self._scan(state), ret_orig(state)
        )

        self.prev_winding_vector = (
            state.cartesian_states[:, [0, 1]] - self.winding_point
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""

        @partial(jax.jit, static_argnums=[1])
        def observation(agent_ind, num_agents):
            # extract scan if exist
            agent_scan = jax.lax.select(
                state.scans is not None, state.scans[agent_ind, :], jnp.zeros((0,))
            )

            # extract states
            agent_state = state.cartesian_states[agent_ind, :]

            # extract relative states
            # (relative_x, relative_y, longitudinal_v, relative_psi)
            relative_states = jax.lax.select(
                num_agents > 1,
                state.cartesian_states[jnp.delete(jnp.arange(num_agents), agent_ind)][
                    [0, 1, 3, 4]
                ]
                - agent_state,
                jnp.zeros((0,)),
            )

            all_states = jnp.hstack((agent_state, relative_states, agent_scan))
            return all_states

        return {a: observation(i, self.num_agents) for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def check_done(self, state: State) -> Tuple[Dict[str, bool], State]:

        winding_vector = state.cartesian_states[:, [0, 1]] - self.winding_point

        # angle differentials, from new winding vectors to previous winding vectors
        winding_angles = jnp.arctan2(
            jnp.cross(winding_vector, self.prev_winding_vector),
            jnp.einsum("ij,ij->i", winding_vector, self.prev_winding_vector),
        )

        self.accumulated_angles = self.accumulated_angles + winding_angles
        self.num_laps = jnp.abs(self.accumulated_angles) / (2 * jnp.pi)
        laps_done = self.num_laps >= self.max_num_laps

        # collision dones
        done_dict = {
            i: (state.collisions[i] or laps_done[i]) for i in range(self.num_agents)
        }

        # update state
        state.num_laps = self.num_laps
        state.done = jnp.array([done_dict[a] for a in self.agents])

        return done_dict, state

    @partial(jax.jit, static_argnums=[0])
    def _scan(self, state: State, key: chex.PRNGKey) -> State:
        get_scan_vmapped = jax.vmap(
            partial(
                get_scan,
                theta_dis=self.theta_dis,
                fov=self.fov,
                num_beams=self.num_beams,
                theta_index_increment=self.theta_index_increment,
                sines=self.scan_sines,
                cosines=self.scan_cosines,
                eps=self.eps,
                orig_x=self.orig_x,
                orig_y=self.orig_y,
                orig_c=self.orig_c,
                orig_s=self.orig_s,
                height=self.height,
                width=self.width,
                resolution=self.resolution,
                dt=self.distance_transform,
                max_range=self.max_range,
            ),
            in_axes=[0],
        )
        scans = get_scan_vmapped(state.cartesian_states[:, [0, 1, 4]])
        noise = jax.random.normal(key, scans.shape) * 0.01
        state.scans = scans + noise
        return state

    @partial(jax.jit, static_argnums=[0])
    def _collisions(self, state: State) -> State:
        # extract vertices from all cars (n_agent, 4, 2)
        all_vertices = jax.vmap(
            partial(get_vertices, length=self.params.length, width=self.params.width),
            in_axes=[0],
        )(state.cartesian_states[:, [0, 1, 4]])

        # check pairwise collisions
        pairwise_indices1, pairwise_indices2 = jnp.triu_indices(self.num_agents, 1)
        pairwise_vertices = jnp.concatenate(
            (all_vertices[pairwise_indices1], all_vertices[pairwise_indices2]), axis=-1
        )
        # (n_agent!, )
        pairwise_collisions = jax.vmap(collision, in_axes=[0])(pairwise_vertices)

        # get indices that are colliding
        collided_ind = jax.lax.select(
            pairwise_collisions,
            jnp.column_stack(pairwise_indices1, pairwise_indices2),
            -1 * jnp.ones((len(pairwise_indices1), 2)),
        ).flatten()
        padded_collisions = jnp.zeros((self.num_agents + 1,))
        padded_collisions.at[collided_ind].set(1)
        padded_collisions = padded_collisions[:-1]

        # check map collisions (n_agent, )
        map_collisions = collision_map(
            state.cartesian_states[:, [0, 1, 4]], self.pixel_centers
        )

        # combine collisions
        full_collisions = jnp.logical_or(padded_collisions, map_collisions)

        # update state
        state.collisions = full_collisions
        return state
