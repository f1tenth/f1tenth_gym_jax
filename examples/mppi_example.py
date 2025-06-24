import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from flax.struct import dataclass
from typing import Callable

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs import F110Env
from f1tenth_gym_jax.envs.dynamic_models import (
    vehicle_dynamics_ks,
    vehicle_dynamics_st_switching,
)
from f1tenth_gym_jax.envs.utils import batchify, unbatchify, Param
from f1tenth_gym_jax.envs.track.cubic_spline import nearest_point_on_trajectory_jax


@dataclass
class MPPIConfig:
    # mppi
    n_iterations: int = 1
    n_steps: int = 10
    n_samples: int = 256
    temperature: float = 0.01
    damping: float = 0.001

    # system
    control_dim: int = 2
    control_limit: jax.Array = jnp.array([[-3.2, -10.0], [3.2, 10.0]])
    dyn_fn: Callable = vehicle_dynamics_ks
    dt: 0.1


@jax.jit
def get_ref_traj(
    predicted_speeds,
    dist_from_segment_start,
    idx,
    waypoints,
    waypoints_distances,
    DT,
):
    total_length = jnp.sum(waypoints_distances)
    s_relative = jnp.concatenate(
        [jnp.array([dist_from_segment_start]), predicted_speeds * DT]
    ).cumsum()
    s_relative = s_relative % total_length
    rolled_distances = jnp.roll(waypoints_distances, -idx)
    wp_dist_cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(rolled_distances)])
    index_relative = jnp.searchsorted(wp_dist_cum, s_relative, side="right") - 1
    index_relative = jnp.clip(index_relative, 0, len(rolled_distances) - 1)
    index_absolute = (idx + index_relative) % (waypoints.shape[0] - 1)
    next_index = (index_absolute + 1) % (waypoints.shape[0] - 1)
    seg_start = wp_dist_cum[index_relative]
    seg_len = rolled_distances[index_relative]
    t = (s_relative - seg_start) / seg_len
    p0 = waypoints[index_absolute][:, 1:3]
    p1 = waypoints[next_index][:, 1:3]
    interpolated_positions = p0 + (p1 - p0) * t[:, jnp.newaxis]
    s0 = waypoints[index_absolute][:, 0]
    s1 = waypoints[next_index][:, 0]
    interpolated_s = (s0 + (s1 - s0) * t) % waypoints[-1, 0]
    yaw0 = waypoints[index_absolute][:, 3]
    yaw1 = waypoints[next_index][:, 3]
    interpolated_yaw = yaw0 + (yaw1 - yaw0) * t
    interpolated_yaw = (interpolated_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    v0 = waypoints[index_absolute][:, 5]
    v1 = waypoints[next_index][:, 5]
    interpolated_speed = v0 + (v1 - v0) * t
    reference = jnp.stack(
        [
            interpolated_positions[:, 0],
            interpolated_positions[:, 1],
            interpolated_speed,
            interpolated_yaw,
            interpolated_s,
            jnp.zeros_like(interpolated_speed),
            jnp.zeros_like(interpolated_speed),
        ],
        axis=1,
    )
    return reference


class MPPI:
    def __init__(
        self,
        config: MPPIConfig,
        env: F110Env,
        rng: jax.random.PRNGKey,
    ):
        self.config = config
        self.env = env
        self.rng = rng

        self.n_iterations = config.n_iterations
        self.n_steps = config.n_steps
        self.n_samples = config.n_samples
        self.a_std = jnp.array(config.control_sample_std)
        self.a_cov_shift = config.a_cov_shift
        self.adaptive_covariance = (
            config.adaptive_covariance and self.n_iterations > 1
        ) or self.a_cov_shift
        self.a_shape = config.control_dim

        self.init_state()
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))

        line = self.env.track.raceline
        self.waypoints = jnp.column_stack((line.xs, line.ys, line.vxs))
        self.waypoint_distances = jnp.linalg.norm(
            self.waypoints[1:, :2] - self.waypoints[:-1, :2], axis=1
        )

    def init_state(self):
        dim_a = jnp.prod(self.a_shape)
        self.rng, _rng = jax.random.split(self.rng)
        self.a_opt = 0.0 * jax.random.uniform(_rng, shape=(self.n_steps, dim_a))

        # a_cov: [n_steps, dim_a, dim_a]
        if self.a_cov_shift:
            self.a_cov = (self.a_std**2) * jnp.tile(
                jnp.eye(dim_a), (self.n_steps, 1, 1)
            )
            self.a_cov_init = self.a_cov.copy()
        else:
            self.a_cov = None
            self.a_cov_init = None

    @partial(jax.jit, static_argnums=(0))
    def weights(self, R):
        # R: [n_samples]
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.config.damping)
        w = jnp.exp(R_stdzd / self.config.temperature)  # [n_samples] np.float32
        w = w / jnp.sum(w)  # [n_samples] np.float32
        return w

    @partial(jax.jit, static_argnums=0)
    def rollout(self, actions, dyn_state):
        def _step_dyn(curr_state, actions):
            next_state = curr_state + self.dyn_fn(curr_state, actions)
            return next_state, next_state

        _, state_traj = jax.lax.scan(_step_dyn, dyn_state, actions)
        return state_traj

    @partial(jax.jit, static_argnums=(0))
    def get_ref(self, state, n_steps=10):
        dist, t, ind = nearest_point_on_trajectory_jax(
            jnp.array([state[0], state[1]]), self.waypoints[:, :2]
        )
        speeds = jnp.ones(n_steps) * self.waypoints[ind, 2]
        reference = get_ref_traj(
            speeds, dist, ind, self.waypoints, self.waypoint_distances, self.config.dt
        )
        return reference, ind

    @partial(jax.jit, static_argnums=0)
    def cost(self, states, reference_traj):
        # states: [n_samples, n_steps, dim_s]
        # reference_traj: [n_steps, dim_s]
        # cost is the squared distance to the reference trajectory
        ref_states = jnp.tile(reference_traj[None, :, :], (self.n_samples, 1, 1))
        cost = jnp.sum((states - ref_states) ** 2, axis=-1)
        return cost  # [n_samples, n_steps]

    @partial(jax.jit, static_argnums=(0))
    def iteration_step(self, a_opt, rng_da, dyn_state):

        # Step 1: sample controls uniformly
        rng_da, rng_da_split1 = jax.random.split(rng_da)
        actions = jax.random.uniform(
            key=rng_da_split1,
            shape=(self.n_samples, self.n_steps, self.a_shape),
            minval=self.config.control_limit[0],
            maxval=self.config.control_limit[1],
        )  # [n_samples, n_steps, dim_a]

        # Step 2: rollout dynamics
        states = jax.vmap(self.rollout, in_axes=(0, None))(actions, dyn_state)

        # Step 3: compute costs
        ref, ind = self.get_ref(dyn_state)
        cost = jax.vmap(self.cost, in_axes=(0, None))(
            states, ref
        )  # [n_samples, n_steps]

        R = jnp.einsum("ij,jj->ij", cost, self.accum_matrix)  # [n_samples, n_steps]
        w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
        a_opt = jax.vmap(jnp.average, (1, None, 1))(actions, 0, w)  # [n_steps, dim_a]

        opt_states = self.rollout(a_opt, dyn_state)

        return a_opt, states, opt_states, rng_da


# TODOs:
# 1. uniform sampling instead of truncated normal
# 2. remove inferenv, use explicit functions for dynamics and rewards
# 3.


def main():
    num_agents = 3
    num_envs = 10
    num_actors = num_agents * num_envs

    env = make(f"Spielberg_{num_agents}_noscan_time_v0")
    line = env.track.raceline
    waypoints = jnp.vstack((line.xs, line.ys, line.vxs)).T

    rng = jax.random.key(0)
    rng2 = jax.random.key(1)

    config = MPPIConfig(
        n_iterations=10,
        n_steps=20,
        n_samples=256,
        control_sample_std=0.1,
        control_limit=jnp.array([[-3.2, -10.0], [3.2, 10.0]]),
        dyn_fn=vehicle_dynamics_ks,
        dt=0.1,
    )

    mppi = MPPI(config, env, rng)

    # Initialize the state
    mppi.init_state()


    def _env_init():
        rng, _rng = jax.random.split(rng2)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        return (env_state, obsv, rng)
    
    def _env_step(runner_state, unused):
        env_state, last_obsv, rng = runner_state
        rng, _rng = jax.random.split(rng)
        step_rngs = jax.random.split(_rng, num_envs)

        # Get the current state of the vehicle
        batched_obs = batchify(last_obsv, env.agents, num_actors)
        batched_actions = jax.vmap(mppi.iteration_step, in_axes=(0, None))(mppi.a_opt, mppi.rng)

        # Unbatch the actions to match the environment's expected input
        env_actions = unbatchify(batched_actions, env.agents, num_envs, num_agents)

        obsv, env_state, _, _, info = jax.vmap(env.step)(step_rngs, env_state, env_actions)
        runner_state = (env_state, obsv, rng)
        return runner_state, runner_state

    # Example of running an iteration step
    a_opt, states, opt_states, rng_da = mppi.iteration_step(
        mppi.a_opt, mppi.rng, env.reset()[1]
    )


if __name__ == "__main__":
    main()