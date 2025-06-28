import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from flax import struct
import chex

from typing import Callable

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs import F110Env
from f1tenth_gym_jax.envs.dynamic_models import vehicle_dynamics_st_switching
from f1tenth_gym_jax.envs.integrator import integrate_rk4
from f1tenth_gym_jax.envs.utils import batchify, unbatchify
from f1tenth_gym_jax.envs.track.cubic_spline import nearest_point_on_trajectory_jax

from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer


@struct.dataclass
class MPPIConfig:
    # mppi
    n_iterations: int = 1
    n_steps: int = 10
    n_samples: int = 128
    temperature: float = 0.01
    damping: float = 0.001
    dt: float = 0.1

    # system
    control_dim: int = 2  # [steering_velocity, longitudinal_acceleration]
    state_dim: int = 7  # [x, y, delta, v, psi, psi_dot, beta]
    dyn_fn: Callable = vehicle_dynamics_st_switching
    int_fn: Callable = integrate_rk4
    control_limit: chex.Array = struct.field(
        default_factory=lambda: jnp.array([[-3.2, -10.0], [3.2, 10.0]])
    )


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
        self.a_shape = config.control_dim

        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))

        line = self.env.track.raceline
        self.waypoints = jnp.column_stack((line.xs, line.ys, line.vxs))
        self.waypoint_distances = jnp.linalg.norm(
            self.waypoints[1:, :2] - self.waypoints[:-1, :2], axis=1
        )

    @partial(jax.jit, static_argnums=(0))
    def weights(self, R):
        # R: [n_samples]
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.config.damping)
        w = jnp.exp(R_stdzd / self.config.temperature)  # [n_samples] np.float32
        w = w / jnp.sum(w)  # [n_samples] np.float32
        return w

    @partial(jax.jit, static_argnums=(0))
    def rollout(self, actions, dyn_state):
        # actions: [n_steps, dim_a]
        # dyn_state: [dim_s]

        def _step_dyn(x, u):
            # x: [dim_s]
            # u: [dim_a]
            x_and_u = jnp.hstack((x, u))
            new_x_and_u = self.config.int_fn(
                self.config.dyn_fn, x_and_u, self.env.params
            )
            next_state = new_x_and_u[: -self.config.control_dim]  # [dim_s]
            return next_state, next_state

        _, state_traj = jax.lax.scan(_step_dyn, dyn_state, actions)
        return state_traj

    @partial(jax.jit, static_argnums=(0, 2))
    def get_ref(self, state, n_steps):
        dist, t, ind = nearest_point_on_trajectory_jax(
            jnp.array([state[0], state[1]]), self.waypoints[:, :2]
        )
        speeds = jnp.ones(n_steps) * self.waypoints[ind, 2]
        reference = get_ref_traj(
            speeds, dist, ind, self.waypoints, self.waypoint_distances, self.config.dt
        )
        return reference, ind

    @partial(jax.jit, static_argnums=(0))
    def cost(self, states, reference_traj):
        # states: [n_samples, n_steps, dim_s]
        # reference_traj: [n_steps, dim_s]
        # cost is the squared distance to the reference trajectory
        # ref_states = jnp.tile(reference_traj[None, :, :], (self.n_samples, 1, 1))
        cost = jnp.sum((states - reference_traj) ** 2, axis=-1)
        return cost  # [n_samples, n_steps]

    @partial(jax.jit, static_argnums=(0))
    def iteration_step(self, rng_da, dyn_state):

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
        ref, ind = self.get_ref(dyn_state, n_steps=self.n_steps - 1)
        cost = jax.vmap(self.cost, in_axes=(0, None))(
            states, ref
        )  # [n_samples, n_steps]
        R = jnp.einsum("ij,jj->ij", cost, self.accum_matrix)  # [n_samples, n_steps]
        w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
        a_opt = jax.vmap(jnp.average, (1, None, 1))(actions, 0, w)  # [n_steps, dim_a]

        opt_states = self.rollout(a_opt, dyn_state)

        return a_opt, states, opt_states, rng_da



def main():
    num_agents = 3
    num_envs = 10
    num_actors = num_agents * num_envs
    num_states = 7

    env = make(f"Spielberg_{num_agents}_noscan_time_v0")

    rng = jax.random.key(0)
    rng2 = jax.random.key(1)

    config = MPPIConfig()

    mppi = MPPI(config, env, rng)

    @jax.jit
    def _env_init():
        rng, _rng, __rng = jax.random.split(rng2, 3)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        dummy_bs = jnp.zeros((num_actors, config.n_samples, config.n_steps, num_states))
        dummy_bos = jnp.zeros((num_actors, config.n_steps, num_states))
        init_rng = jax.random.split(__rng, num_actors)
        return (env_state, obsv, dummy_bs, dummy_bos, init_rng, rng)

    @jax.jit
    def _env_step(runner_state, unused):
        (
            env_state,
            last_obsv,
            last_batched_states,
            last_batched_opt_states,
            last_batched_rng,
            rng,
        ) = runner_state
        rng, _rng = jax.random.split(rng)
        step_rngs = jax.random.split(_rng, num_envs)

        # Get the current state of the vehicle
        batched_obs = batchify(last_obsv, env.agents, num_actors)
        dyn_states = batched_obs[..., :7]

        # batched_actions [num_actors, num_steps, dim_a]
        # batched_states [num_actors, num_samples, num_steps, dim_s]
        # batched_opt_states [num_actors, num_steps, dim_s]
        batched_actions, batched_states, batched_opt_states, batched_rng = jax.vmap(
            mppi.iteration_step, in_axes=(0, 0)
        )(last_batched_rng, dyn_states)

        current_action = batched_actions[:, 0, :]
        # Unbatch the actions to match the environment's expected input
        env_actions = unbatchify(current_action, env.agents, num_envs, num_agents)

        obsv, env_state, _, _, info = jax.vmap(env.step)(
            step_rngs, env_state, env_actions
        )
        runner_state = (
            env_state,
            obsv,
            batched_states,
            batched_opt_states,
            batched_rng,
            rng,
        )
        return runner_state, runner_state

    final_runner, all_runner_state = jax.lax.scan(_env_step, _env_init(), length=1000)

    player = TrajRenderer(env)
    player.render(np.array(all_runner_state[0].cartesian_states))


if __name__ == "__main__":
    main()
