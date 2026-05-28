import os
import time

os.environ["QT_QPA_PLATFORM"] = os.environ.get("QT_QPA_PLATFORM", "offscreen")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer


def rollout(num_steps: int | None = None):
    """
    Roll out the JAX environment and return a trajectory array.
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


def record_gif(
    num_steps: int | None = None,
    output: pathlib.Path = pathlib.Path("f1tenth_gym_jax_rollout.gif"),
) -> int:
    """
    Roll out the JAX environment and save a short rendered GIF.
    """
    output = pathlib.Path(output)
    start = time.time()
    env, trajectory = rollout(num_steps=num_steps)
    renderer = TrajRenderer(env, render_mode="rgb_array")
    try:
        frames = []
        for _ in range(trajectory.shape[0]):
            frame = renderer.render(trajectory)
            if frame is not None:
                frames.append(Image.fromarray(frame))
    finally:
        renderer.close()

    if frames:
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 * env.params.timestep * env.params.timestep_ratio),
            loop=0,
        )
    print(f"Saved {len(frames)} frames to {output} in {time.time() - start:.2f}s")
    return len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=None, help="Maximum rollout steps."
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("f1tenth_gym_jax_rollout.gif"),
        help="Output GIF path.",
    )
    args = parser.parse_args()

    record_gif(num_steps=args.steps, output=args.output)


if __name__ == "__main__":
    main()
