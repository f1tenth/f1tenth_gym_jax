import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.rendering import WebRenderer
from f1tenth_gym_jax.envs.utils import unbatchify


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


@jax.jit
def pure_pursuit_2(pose, wpts, Ld=3.0, L=0.33, dt=0.1):
    x, y, theta, current_steer, current_v = pose
    dx, dy = wpts[:, 0] - x, wpts[:, 1] - y
    xv = dx * jnp.cos(theta) + dy * jnp.sin(theta)
    yv = -dx * jnp.sin(theta) + dy * jnp.cos(theta)

    ds = jnp.hypot(xv, yv)
    i = jnp.argmin(jnp.abs(ds - Ld))
    _, y_ld = xv[i], yv[i]

    kappa = 2.0 * y_ld / (Ld**2)
    desired_steer = jnp.arctan(L * kappa)

    sv = (desired_steer - current_steer) / dt
    sv = jnp.clip(sv, -3.2, 3.2)

    desired_v = wpts[i, 2]
    accl = (desired_v - current_v) / dt
    accl = jnp.clip(accl, -10.0, 10.0)

    return jnp.array([sv, accl])


def run_waypoint_follow(
    num_agents: int = 3,
    num_envs: int = 10,
    num_steps: int = 1000,
    render: bool = True,
    render_output: pathlib.Path = pathlib.Path("f1tenth_gym_jax_rollout.html"),
):
    _validate_positive_int("num_agents", num_agents)
    _validate_positive_int("num_envs", num_envs)
    _validate_positive_int("num_steps", num_steps)

    num_actors = num_agents * num_envs

    env = make(
        f"Spielberg_{num_agents}_noscan_nocollision_progress_acceleration+steeringvelocity_1_v0"
    )
    raceline = env.track.raceline
    waypoints = jnp.vstack((raceline.xs, raceline.ys, raceline.vxs)).T
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

        batched_states = env_state.cartesian_states.reshape((num_actors, -1))
        batched_actions = jax.vmap(pure_pursuit_2, in_axes=(0, None))(
            batched_states[:, [0, 1, 4, 2, 3]], waypoints
        )
        env_actions = unbatchify(batched_actions, env.agents, num_envs, num_agents)

        obsv, env_state, _, _, info = jax.vmap(env.step)(
            step_rngs, env_state, env_actions
        )
        runner_state = (env_state, obsv, rng)
        return runner_state, runner_state

    final_runner, all_runner_state = jax.lax.scan(
        env_step, env_init(rng), length=num_steps
    )

    if render:
        WebRenderer(env).render(
            np.array(all_runner_state[0].cartesian_states),
            output_path=render_output,
        )
    return final_runner, all_runner_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--no-render", action="store_true", help="Skip trajectory rendering."
    )
    parser.add_argument(
        "--render-output",
        type=pathlib.Path,
        default=pathlib.Path("f1tenth_gym_jax_rollout.html"),
        help="Output HTML dashboard path.",
    )
    args = parser.parse_args()

    run_waypoint_follow(
        num_agents=args.num_agents,
        num_envs=args.num_envs,
        num_steps=args.steps,
        render=not args.no_render,
        render_output=args.render_output,
    )


if __name__ == "__main__":
    main()
