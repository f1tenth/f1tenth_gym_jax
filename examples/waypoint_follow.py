import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.utils import batchify, unbatchify
from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer

from f1tenth_gym_jax.envs.track.cubic_spline import (
    nearest_point_on_trajectory_jax,
    first_point_on_trajectory,
)


@jax.jit
def pure_pursuit(pose, waypoints, lookahead_distance=0.8, wheelbase=0.33):
    nearest_dist, t, i = nearest_point_on_trajectory_jax(pose[:2], waypoints[:, :2])
    i2, lookahead_point, dist2 = first_point_on_trajectory(
        pose[:2], waypoints[:, :2], lookahead_distance
    )

    waypoint_y = jnp.dot(
        jnp.array([jnp.sin(-pose[2]), jnp.cos(-pose[2])]),
        lookahead_point - pose[:2],
    )
    velocity = waypoints[i, 2]
    radius = 1.0 / (2.0 * waypoint_y / lookahead_distance**2)
    steering = jnp.arctan(wheelbase/radius)
    steering = jax.lax.select(jnp.abs(waypoint_y) < 1e-6, 0.0, steering)
    # jax.debug.print(
    #     "Pure Pursuit: current_pose={pose}, nearest_dist={nearest_dist}, t={t}, i={i}, "
    #     "lookahead_point={lookahead_point}, waypoint_y={waypoint_y}, "
    #     "velocity={velocity}, radius={radius}, steering={steering}",
    #     pose=pose,
    #     nearest_dist=nearest_dist,
    #     t=t,
    #     i=i,
    #     lookahead_point=lookahead_point,
    #     waypoint_y=waypoint_y,
    #     velocity=velocity,
    #     radius=radius,
    #     steering=steering,
    # )
    return jnp.array([steering, velocity])


def main():

    num_agents = 3
    num_envs = 10
    num_actors = num_agents * num_envs

    env = make(f"Monza_{num_agents}_noscan_time_v0")
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
