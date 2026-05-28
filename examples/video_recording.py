import os
import time

os.environ["QT_QPA_PLATFORM"] = os.environ.get("QT_QPA_PLATFORM", "offscreen")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer


def main():
    """
    Roll out the JAX environment and save a short rendered GIF.
    """
    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_60_v0"
    )
    rng = jax.random.key(0)
    _, state = env.reset(rng)

    trajectory = []
    actions = {"agent_0": jnp.array([0.0, 1.0])}

    start = time.time()
    for _ in range(env.params.max_steps):
        rng, step_rng = jax.random.split(rng)
        _, state, _, dones, _ = env.step(step_rng, state, actions)
        trajectory.append(np.asarray(state.cartesian_states))
        if bool(dones["__all__"]):
            break

    trajectory = np.asarray(trajectory)[:, None, :, :]
    renderer = TrajRenderer(env, render_mode="rgb_array")
    try:
        frames = []
        for _ in range(trajectory.shape[0]):
            frame = renderer.render(trajectory)
            if frame is not None:
                frames.append(Image.fromarray(frame))
    finally:
        renderer.close()

    output = "f1tenth_gym_jax_rollout.gif"
    if frames:
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 * env.params.timestep * env.params.timestep_ratio),
            loop=0,
        )
    print(f"Saved {len(frames)} frames to {output} in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
