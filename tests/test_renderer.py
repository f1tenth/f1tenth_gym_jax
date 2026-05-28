import json
import pathlib
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.rendering import TrajRenderer, WebRenderer


def _payload_from_dashboard(path: pathlib.Path) -> dict:
    html = path.read_text()
    marker = '<script id="rollout-data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


class TestRenderer(unittest.TestCase):
    def _make_renderer_inputs(self, max_steps=5):
        env = make(
            f"Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_{max_steps}_v0"
        )
        _, state = env.reset(jax.random.key(0))
        action = {"agent_0": jnp.array([0.0, 1.0])}
        _, next_state, _, _, _ = env.step(jax.random.key(1), state, action)
        trajectory = np.stack(
            [
                np.asarray(state.cartesian_states),
                np.asarray(next_state.cartesian_states),
            ]
        )[:, None, :, :]
        return env, trajectory

    def test_traj_renderer_aliases_web_renderer(self):
        self.assertIs(TrajRenderer, WebRenderer)

    def test_web_dashboard_render_smoke(self):
        env, trajectory = self._make_renderer_inputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "rollout.html"
            rendered = WebRenderer(env).render(trajectory, output_path=output)

            self.assertEqual(rendered, output)
            self.assertTrue(output.exists())
            html = output.read_text()
            self.assertIn("Batched Rollout Overview", html)
            self.assertIn("Timestep scrubber", html)
            self.assertIn("Speed multiplier", html)
            self.assertIn("actual real time", html)
            self.assertIn("Visualization Options", html)
            self.assertIn("cameraSelect", html)
            self.assertIn("bindCanvasPanZoom", html)
            self.assertIn("overviewOtherRollouts", html)
            self.assertIn("playbackFullTrace", html)
            self.assertIn("Artifact overlays", html)
            self.assertNotIn(' + "<span><i class="swatch"', html)

            payload = _payload_from_dashboard(output)
            self.assertEqual(payload["summary"]["rollouts"], 1)
            self.assertEqual(payload["summary"]["steps"], 2)
            self.assertEqual(payload["summary"]["agents"], 1)
            self.assertEqual(payload["env"]["agents"], ["agent_0"])
            self.assertEqual(
                len(payload["track"]["centerline"]), len(env.track.centerline.xs)
            )
            self.assertTrue(
                payload["map"]["image"].startswith("data:image/png;base64,")
            )
            self.assertEqual(payload["artifacts"]["overlays"], [])

    def test_web_dashboard_accepts_artifact_overlays(self):
        env, trajectory = self._make_renderer_inputs()
        path_overlay = np.zeros((2, 1, 1, 3, 2))
        path_overlay[:, 0, 0, :, 0] = np.array([[0.0, 0.4, 0.8], [0.1, 0.5, 0.9]])
        path_overlay[:, 0, 0, :, 1] = np.array([[0.0, 0.1, 0.0], [0.2, 0.3, 0.2]])
        sample_overlay = np.repeat(path_overlay[:, :, :, None, :, :], 2, axis=3)
        sample_values = np.ones(sample_overlay.shape[:-1])
        sample_values[..., 1, :] *= 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "artifacts.html"
            WebRenderer(env).render(
                trajectory,
                output_path=output,
                artifacts={
                    "overlays": [
                        {
                            "id": "selected",
                            "label": "selected path",
                            "type": "paths",
                            "points": path_overlay,
                            "color": "#ff0000",
                        },
                        {
                            "id": "samples",
                            "label": "sample paths",
                            "type": "sample_paths",
                            "points": sample_overlay,
                            "values": sample_values,
                            "value_label": "cost",
                        },
                    ]
                },
            )

            html = output.read_text()
            self.assertIn("drawArtifactOverlays", html)
            payload = _payload_from_dashboard(output)
            overlays = payload["artifacts"]["overlays"]
            self.assertEqual(
                [overlay["id"] for overlay in overlays], ["selected", "samples"]
            )
            self.assertEqual(overlays[0]["type"], "paths")
            self.assertEqual(overlays[1]["type"], "sample_paths")
            self.assertEqual(overlays[1]["valueLabel"], "cost")
            self.assertEqual(overlays[1]["valueMin"], 1.0)
            self.assertEqual(overlays[1]["valueMax"], 2.0)

    def test_web_dashboard_preserves_batched_rollouts(self):
        env, trajectory = self._make_renderer_inputs()
        batched_trajectory = np.repeat(trajectory, 2, axis=1)
        batched_trajectory[:, 1, :, 0] += 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "batched.html"
            WebRenderer(env).render(batched_trajectory, output_path=output)

            payload = _payload_from_dashboard(output)
            self.assertEqual(payload["summary"]["rollouts"], 2)
            self.assertEqual(len(payload["trajectory"]), 2)
            self.assertEqual(len(payload["rolloutStats"]), 2)

    def test_default_layout_preserves_short_step_major_batches(self):
        env, trajectory = self._make_renderer_inputs()
        short_batch = np.repeat(trajectory[:1], 3, axis=1)
        short_batch[:, 1, :, 0] += 1.0
        short_batch[:, 2, :, 0] += 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "short-batch.html"
            WebRenderer(env).render(short_batch, output_path=output)

            payload = _payload_from_dashboard(output)
            self.assertEqual(payload["summary"]["rollouts"], 3)
            self.assertEqual(payload["summary"]["steps"], 1)
            self.assertEqual(len(payload["trajectory"]), 3)

    def test_web_dashboard_accepts_batch_major_layout(self):
        env, trajectory = self._make_renderer_inputs()
        batch_major = np.transpose(trajectory, (1, 0, 2, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "batch-major.html"
            WebRenderer(env, trajectory_layout="batch_major").render(
                batch_major,
                output_path=output,
            )

            payload = _payload_from_dashboard(output)
            self.assertEqual(payload["summary"]["rollouts"], 1)
            self.assertEqual(payload["summary"]["steps"], 2)

    def test_desktop_render_modes_are_removed(self):
        env, _ = self._make_renderer_inputs()

        for render_mode in ("human", "rgb_array"):
            with self.subTest(render_mode=render_mode):
                with self.assertRaisesRegex(ValueError, "render_mode='html'"):
                    WebRenderer(env, render_mode=render_mode)

    def test_renderer_rejects_invalid_playback_parameters(self):
        env, _ = self._make_renderer_inputs()

        invalid_kwargs = [
            {"render_fps": 0},
            {"render_fps": -1},
            {"window_width": 0},
            {"window_height": 0},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    WebRenderer(env, **kwargs)

    def test_play_pause_tracks_default_playback_state(self):
        env, trajectory = self._make_renderer_inputs()
        renderer = WebRenderer(env)

        self.assertTrue(renderer.playing)
        renderer.play_pause()
        self.assertFalse(renderer.playing)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "paused.html"
            renderer.render(trajectory, output_path=output)
            payload = _payload_from_dashboard(output)

            self.assertFalse(payload["playing"])
            self.assertIn("let playing = Boolean(payload.playing);", output.read_text())

        renderer.play_pause()
        self.assertTrue(renderer.playing)
