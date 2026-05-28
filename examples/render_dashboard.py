import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.rendering import WebRenderer


def rollout(num_steps: int | None = None):
    """
    Roll out the JAX environment and return a trajectory array for dashboard rendering.
    """
    if num_steps is not None and num_steps < 1:
        raise ValueError("num_steps must be positive when provided.")

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_60_v0"
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

    return env, np.asarray(trajectory)[:, None, :, :]


def save_dashboard(
    num_steps: int | None = None,
    output: pathlib.Path = pathlib.Path("f1tenth_gym_jax_rollout.html"),
) -> pathlib.Path:
    """
    Roll out the JAX environment and save a self-contained web dashboard.
    """
    output = pathlib.Path(output)
    env, trajectory = rollout(num_steps=num_steps)
    return WebRenderer(env).render(trajectory, output_path=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=None, help="Maximum rollout steps."
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("f1tenth_gym_jax_rollout.html"),
        help="Output HTML dashboard path.",
    )
    args = parser.parse_args()

    output = save_dashboard(num_steps=args.steps, output=args.output)
    print(f"Saved rollout dashboard to {output}")


if __name__ == "__main__":
    main()
