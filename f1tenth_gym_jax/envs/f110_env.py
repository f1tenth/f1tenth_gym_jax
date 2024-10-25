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

# numpy
import numpy as np

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

        # set pixel centers of occupancy map
        self._set_pixelcenters()

        # TODO: keep all start line, lap information in frenet frame

        # reset modes
        self.reset_fn = make_reset_fn(
            **self.config["reset_config"], track=self.track, num_agents=self.num_agents
        )

        # # start line info TODO: check if still needed
        # self.start_xs = np.zeros((self.num_agents,))
        # self.start_ys = np.zeros((self.num_agents,))
        # self.start_thetas = np.zeros((self.num_agents,))
        # self.start_rot = np.eye(2)
        # # initiate stuff
        # self.sim = Simulator(
        #     self.params,
        #     self.num_agents,
        #     self.seed,
        #     time_step=self.timestep,
        #     integrator=self.integrator,
        #     model=self.model,
        #     action_type=self.action_type,
        # )
        # self.sim.set_map(self.map)

        # # observations
        # self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        # assert (
        #     "type" in self.observation_config
        # ), "observation_config must contain 'type' key"
        # self.observation_type = observation_factory(env=self, **self.observation_config)
        # self.observation_space = self.observation_type.space()

        # # action space
        # self.action_space = from_single_to_multi_action_space(
        #     self.action_type.space, self.num_agents
        # )

    # @partial(jax.jit, static_argnums=(0,))
    # def step(
    #     self,
    #     key: chex.PRNGKey,
    #     state: State,
    #     actions: Dict[str, chex.Array],
    #     reset_state: Optional[State] = None,
    # ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
    #     """Performs step transitions in the environment. Resets the environment if done.
    #     To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

    #     key, key_reset = jax.random.split(key)
    #     obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

    #     if reset_state is None:
    #         obs_re, states_re = self.reset(key_reset)
    #     else:
    #         states_re = reset_state
    #         obs_re = self.get_obs(states_re)

    #     # Auto-reset environment based on termination
    #     states = jax.tree_map(
    #         lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
    #     )
    #     obs = jax.tree_map(
    #         lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
    #     )
    #     return obs, states, rewards, dones, infos

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
    def check_done(self, state: State) -> Dict[str, bool]:
        # TODO: check current s, ey

        # TODO:

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return bool(done), self.toggle_list >= 4

    @partial(jax.jit, static_argnums=[0])
    def _scan(self, state: State, key: chex.PRNGKey) -> State:
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
