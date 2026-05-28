import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make


class TestRenderer(unittest.TestCase):
    def _make_renderer_inputs(self, max_steps=5):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        try:
            from f1tenth_gym_jax.envs.rendering.renderer import TrajRenderer
        except Exception as exc:
            self.skipTest(f"Renderer dependencies are unavailable: {exc}")

        env = make(
            f"Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_{max_steps}_v0"
        )
        return env, TrajRenderer

    def test_rgb_array_render_smoke(self):
        env, TrajRenderer = self._make_renderer_inputs()

        renderer = TrajRenderer(env, render_mode="rgb_array")
        try:
            _, state = env.reset(jax.random.key(0))
            trajectory = np.asarray(state.cartesian_states)[None, None, :, :]
            frame = renderer.render(trajectory)
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 3)
        finally:
            renderer.close()

    def test_rgb_array_render_reuses_vehicle_items(self):
        env, TrajRenderer = self._make_renderer_inputs(max_steps=2)
        _, state = env.reset(jax.random.key(0))
        action = {"agent_0": jnp.array([0.0, 1.0])}
        _, next_state, _, _, _ = env.step(jax.random.key(1), state, action)
        trajectory = np.stack(
            [
                np.asarray(state.cartesian_states),
                np.asarray(next_state.cartesian_states),
            ]
        )[:, None, :, :]

        renderer = TrajRenderer(env, render_mode="rgb_array")
        try:
            self.assertIsInstance(renderer.render(trajectory), np.ndarray)
            items_after_first = len(renderer.canvas.getPlotItem().listDataItems())

            self.assertIsInstance(renderer.render(trajectory), np.ndarray)
            items_after_second = len(renderer.canvas.getPlotItem().listDataItems())

            self.assertEqual(items_after_first, items_after_second)
            self.assertIsNone(renderer.render(trajectory))
        finally:
            renderer.close()
