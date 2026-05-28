import os
import unittest

import jax
import numpy as np

from f1tenth_gym_jax import make


class TestRenderer(unittest.TestCase):
    def test_rgb_array_render_smoke(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        try:
            from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer
        except Exception as exc:
            self.skipTest(f"Renderer dependencies are unavailable: {exc}")

        env = make(
            "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        _, state = env.reset(jax.random.key(0))
        trajectory = np.asarray(state.cartesian_states)[None, None, :, :]

        renderer = TrajRenderer(env, render_mode="rgb_array")
        try:
            frame = renderer.render(trajectory)
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 3)
        finally:
            renderer.close()
