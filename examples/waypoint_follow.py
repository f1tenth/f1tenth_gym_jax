import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.utils import batchify, unbatchify
from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer


@jax.jit
def wrap_to_pi(angle):
    """Wrap to [-π, π)."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


@jax.jit
def pure_pursuit(pose, waypoints, lookahead_distance=0.8, wheelbase=0.33):
    dx = waypoints[:, 0] - pose[0]
    dy = waypoints[:, 1] - pose[1]
    dists = jnp.hypot(dx, dy)
    closest_idx = jnp.argmin(dists)

    bearings = jnp.arctan2(dy, dx)
    rel_angles = bearings - pose[2]

    mask = (dists > lookahead_distance) & (jnp.abs(rel_angles) <= jnp.pi / 2)
    idxs = jnp.arange(waypoints.shape[0])
    valid_idxs = jnp.where(mask, idxs, closest_idx)
    idx_target = jnp.min(valid_idxs)
    x_t, y_t, _ = waypoints[idx_target]
    alpha = wrap_to_pi(jnp.arctan2(y_t - pose[1], x_t - pose[0]) - pose[2])
    steering = jnp.arctan2(2 * wheelbase * jnp.sin(alpha), lookahead_distance)
    velocity = waypoints[closest_idx, 2]
    return jnp.array([steering, velocity])


def main():

    num_agents = 3
    num_envs = 10
    num_actors = num_agents * num_envs

    env = make(f"Spielberg_{num_agents}_noscan_time_v0")
    l = env.track.raceline
    waypoints = jnp.vstack((l.xs, l.ys, l.vxs)).T

    rng = jax.random.key(0)

    @jax.jit
    def env_init(rng):
        rng, _rng = jax.random.split(rng)
        reset_rngs = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rngs)
        return (env_state, obsv, rng)

    @jax.jit
    def env_step(runner_state, unused):
        env_state, last_obsv, rng = runner_state
        rng, _rng = jax.random.split(rng)
        step_rngs = jax.random.split(_rng, num_envs)

        batched_obs = batchify(last_obsv, env.agents, num_actors)
        batched_actions = jax.vmap(pure_pursuit, in_axes=(0, None))(
            batched_obs[:, [0, 1, 4]], waypoints
        )
        env_actions = unbatchify(batched_actions, env.agents, num_envs, num_agents)

        obsv, env_state, _, _, info = jax.vmap(env.step)(
            step_rngs, env_state, env_actions
        )
        runner_state = (env_state, obsv, rng)
        return runner_state, runner_state

    final_runner, all_runner_state = jax.lax.scan(env_step, env_init(rng), length=1000)

    player = TrajRenderer(env)
    player.render(np.array(all_runner_state[0].cartesian_states))
    print(jnp.any(all_runner_state[0].collisions))


if __name__ == "__main__":
    main()
