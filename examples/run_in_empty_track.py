import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.rendering import WebRenderer


def rollout(
    num_steps: int | None = None,
    render: bool = True,
    render_output: pathlib.Path = pathlib.Path("f1tenth_gym_jax_rollout.html"),
):
    """
    Run a simple collision-free rollout with the current JAX environment API.
    """
    if num_steps is not None and num_steps < 1:
        raise ValueError("num_steps must be positive when provided.")

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_100_v0"
    )
    rng = jax.random.key(0)
    _, state = env.reset(rng)

    trajectory = []
    actions = {"agent_0": jnp.array([0.0, 1.0])}
    max_steps = (
        env.params.max_steps
        if num_steps is None
        else min(num_steps, env.params.max_steps)
    )

    for _ in range(max_steps):
        rng, step_rng = jax.random.split(rng)
        _, state, _, dones, _ = env.step(step_rng, state, actions)
        trajectory.append(np.asarray(state.cartesian_states))
        if bool(dones["__all__"]):
            break

    trajectory = np.asarray(trajectory)[:, None, :, :]
    if render:
        WebRenderer(env).render(trajectory, output_path=render_output)
    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=None, help="Maximum rollout steps."
    )
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

    rollout(
        num_steps=args.steps,
        render=not args.no_render,
        render_output=args.render_output,
    )


if __name__ == "__main__":
    main()
