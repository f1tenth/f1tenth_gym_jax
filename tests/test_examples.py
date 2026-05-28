import importlib.util
import json
import pathlib
import tempfile
import unittest

import jax
import numpy as np
import yaml

from f1tenth_gym_jax import make


def _load_example_module(name: str):
    path = pathlib.Path(__file__).parent.parent / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExamples(unittest.TestCase):
    def test_notebook_examples_use_current_jax_environment_api(self):
        examples_dir = pathlib.Path(__file__).parent.parent / "examples"
        banned_fragments = (
            "f110_gym",
            "gym.make",
            "gymnasium",
            "jax.random.PRNGKey",
        )

        for notebook_name in ("benchmark_example.ipynb", "rendering_example.ipynb"):
            with self.subTest(notebook=notebook_name):
                notebook = json.loads((examples_dir / notebook_name).read_text())
                source = "".join(
                    line
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                    for line in cell["source"]
                )

                self.assertIn("from f1tenth_gym_jax import make", source)
                self.assertIn("jax.random.key", source)
                for fragment in banned_fragments:
                    self.assertNotIn(fragment, source)

                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])

    def test_mppi_waypoint_geometry_uses_raceline_arclength_and_xy(self):
        mppi_example = _load_example_module("mppi_example")
        env = make(
            "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        config = mppi_example.MPPIConfig(n_samples=2, n_steps=2)

        mppi = mppi_example.MPPI(config, env, jax.random.key(0))

        expected_xy = np.column_stack((env.track.raceline.xs, env.track.raceline.ys))
        expected_distances = np.linalg.norm(expected_xy[1:] - expected_xy[:-1], axis=1)
        np.testing.assert_allclose(
            np.asarray(mppi.waypoints[:, 0]), env.track.raceline.s
        )
        np.testing.assert_allclose(
            np.asarray(mppi.waypoint_distances),
            expected_distances,
            rtol=1e-6,
            atol=1e-5,
        )

    def test_video_recording_writes_requested_gif(self):
        video_recording = _load_example_module("video_recording")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "rollout.gif"
            frames = video_recording.record_gif(num_steps=1, output=output)

            self.assertEqual(frames, 1)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_random_trackgen_uses_yaml_resolution_for_centerline_scale(self):
        random_trackgen = _load_example_module("random_trackgen")
        track = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
                [0.0, 0.0],
            ]
        )
        track_int = track + np.array([1.0, 1.0])
        track_ext = track - np.array([1.0, 1.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = pathlib.Path(tmpdir)
            random_trackgen.convert_track(
                track, track_int, track_ext, "RandomTrack0", outdir
            )

            track_dir = outdir / "RandomTrack0"
            with (track_dir / "RandomTrack0.yaml").open() as yaml_file:
                metadata = yaml.safe_load(yaml_file)
            centerline = np.loadtxt(
                track_dir / "RandomTrack0_centerline.csv",
                delimiter=",",
                comments="#",
            )

            self.assertEqual(metadata["resolution"], random_trackgen.MAP_RESOLUTION)
            self.assertTrue((track_dir / metadata["image"]).exists())
            np.testing.assert_allclose(centerline[0, :2], np.array([0.0, 0.0]))
            np.testing.assert_allclose(
                centerline[:, 2:],
                random_trackgen.TRACK_WIDTH * random_trackgen.MAP_RESOLUTION,
            )
