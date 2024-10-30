"""
Jax jittable f1tenth_gym environment

Based on JaxMARL api which follows PettingZoo/Gymnax

Author: Hongrui Zheng
"""

# typing
from typing import Dict, Tuple

from .multi_agent_env import MultiAgentEnv
from .spaces import Box

# jax
import jax
import jax.numpy as jnp
import chex

# numpy scipy
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

# other
from functools import partial

# dataclasses
from .utils import State, Param

# track
from .track import Track

# dynamics
from .dynamic_models import vehicle_dynamics_ks, vehicle_dynamics_st

# integrators
from .integrator import integrate_euler, integrate_rk4

# scanning
from jax_pf.ray_marching import get_scan

# collisions
from .collision_models import collision, collision_map, get_vertices


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
        super().__init__(num_agents=num_agents)
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
                (self.state_size + self.all_other_state_size + self.scan_size,),
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
                    self.state_size + self.all_other_state_size,
                    self.state_size + self.all_other_state_size + self.scan_size,
                )
            ),
        }

        # load map
        self.track = Track.from_track_name(params.map_name)

        # get a interior point of track as winding number looking point
        start_point_curvature = self.track.centerline.calc_curvature(0.0)
        self.winding_point = jnp.array(
            self.track.frenet_to_cartesian(
                s=0.0, ey=np.sign(start_point_curvature) * 1.5, ephi=0.0
            )
        )[:2]

        # set pixel centers of occupancy map
        self._set_pixelcenters()

        # TODO: keep all start line, lap information in frenet frame

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
        integrator = jax.vmap(self.integrator_func, in_axes=[None, 0, None])
        new_x_and_u = integrator(self.model_func, x_and_u, self.params)
        state = state.replace(cartesian_states=new_x_and_u[:, :-2])
        state = jax.lax.cond(
            self.params.produce_scans, self._scan, self._ret_orig_state, state, key
        )

        # 2. collisions
        state = self._collisions(state)

        # 2. get obs
        obs = self.get_obs(state)

        # 3. dones
        dones = self.check_done(state)

        # 4. rewards
        rewards = self.get_reward(state)

        # 5. info
        infos = {}

        return obs, state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        # reset lap counters etc
        # self.num_laps = jnp.zeros((self.num_agents,), dtype=int)
        # self.accumulated_angles = jnp.zeros(
        #     self.num_agents,
        # )

        # reset states
        s_key, ey_key = jax.random.split(key)
        # randomly choose first agent location [0, 1] on entire arc length
        first_agent_s_loc = jax.random.uniform(s_key)
        first_agent_s = first_agent_s_loc * self.track.length
        first_agent_ey = jax.random.uniform(ey_key, minval=-0.3, maxval=0.3)
        # set up following agents in a grid pattern
        s_locs = jnp.linspace(
            first_agent_s,
            first_agent_s + 0.5 * (self.num_agents - 1),
            self.num_agents,
            endpoint=True,
        )
        ey_locs = first_agent_ey * jnp.where(
            jnp.arange(self.num_agents) % 2 == 0, 1, -1
        )
        ephi_locs = jnp.zeros((self.num_agents,))
        initial_states_frenet = jnp.column_stack((s_locs, ey_locs, ephi_locs))
        initial_poses = self.track.vmap_frenet_to_cartesian_jax(initial_states_frenet)
        initial_states = jnp.zeros((self.num_agents, self.state_size))
        initial_states = initial_states.at[:, [0, 1, 4]].set(initial_poses)

        state = State(
            rewards=jnp.zeros((self.num_agents,)),
            done=jnp.full((self.num_agents), False),
            step=0,
            cartesian_states=initial_states,
            frenet_states=initial_states_frenet,
            num_laps=jnp.full((self.num_agents), 0),
            collisions=jnp.zeros((self.num_agents,), dtype=bool),
            scans=jnp.zeros((self.num_agents, self.num_beams)),
            prev_winding_vector=jnp.zeros((self.num_agents, 2)),
            accumulated_angles=jnp.zeros((self.num_agents, 1)),
        )

        # scan if needed
        state = jax.lax.cond(
            self.params.produce_scans, self._scan, self._ret_orig_state, state, key
        )

        # reset winding vector
        state = state.replace(prev_winding_vector = (
            state.cartesian_states[:, [0, 1]] - self.winding_point
        ))
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""

        @partial(jax.jit, static_argnums=[1])
        def observation(agent_ind, num_agents):
            # extract scan
            agent_scan = state.scans[agent_ind, :]

            # extract states
            agent_state = state.cartesian_states[agent_ind, :]

            # extract relative states
            # (relative_x, relative_y, longitudinal_v, relative_psi)
            # this should automatically deal with only one agent
            other_agent_indices = jnp.delete(
                jnp.arange(num_agents), agent_ind, assume_unique_indices=True
            )
            other_agent_poses = state.cartesian_states[other_agent_indices, :][
                :, jnp.array([0, 1, 4])
            ]
            relative_poses = other_agent_poses - agent_state[jnp.array([0, 1, 4])]
            other_agent_velocities = state.cartesian_states[other_agent_indices, 3]
            relative_states = jnp.column_stack(
                (
                    relative_poses[:, 0],
                    relative_poses[:, 1],
                    other_agent_velocities,
                    relative_poses[:, 2],
                )
            ).flatten()

            all_states = jnp.hstack((agent_state, relative_states, agent_scan))
            return all_states

        return {a: observation(i, self.num_agents) for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def check_done(self, state: State) -> Tuple[Dict[str, bool], State]:

        winding_vector = state.cartesian_states[:, [0, 1]] - self.winding_point

        # angle differentials, from new winding vectors to previous winding vectors
        winding_angles = jnp.arctan2(
            jnp.cross(winding_vector, state.prev_winding_vector),
            jnp.einsum("ij,ij->i", winding_vector, state.prev_winding_vector),
        )

        state = state.replace(accumulated_angles = state.accumulated_angles + winding_angles)
        state = state.replace(num_laps = jnp.abs(state.accumulated_angles) / (2 * jnp.pi))
        laps_done = state.num_laps >= self.params.max_num_laps

        # collision dones
        done_dict = {
            a: jnp.logical_or(state.collisions[i], laps_done[i]) for i, a in enumerate(self.agents)
        }

        # update state
        state = state.replace(done = jnp.array([done_dict[a] for a in self.agents]))

        return done_dict, state
    
    @partial(jax.jit, static_argnums=[0])
    def get_reward(self, state: State) -> Dict[str, float]:
        def reward(i):
            return 0.0
        return {a: reward(i) for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def _ret_orig_state(self, state: State, key: chex.PRNGKey) -> State:
        new_state = state.replace(scans=jnp.zeros((self.num_agents, self.num_beams)))
        return new_state

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
        new_state = state.replace(scans=scans + noise)
        return new_state

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
            jnp.column_stack((pairwise_collisions, pairwise_collisions)),
            jnp.column_stack((pairwise_indices1, pairwise_indices2)),
            -1 * jnp.ones((len(pairwise_indices1), 2), dtype=int),
        ).flatten()
        padded_collisions = jnp.zeros((self.num_agents + 1,))
        padded_collisions.at[collided_ind].set(1)
        padded_collisions = padded_collisions[:-1]

        # check map collisions (n_agent, )
        map_collisions = collision_map(all_vertices, self.pixel_centers)

        # combine collisions
        full_collisions = jnp.logical_or(padded_collisions, map_collisions)

        # update state
        state = state.replace(collisions = full_collisions)
        return state
