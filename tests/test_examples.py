import importlib.util
import pathlib
import unittest

import jax
import numpy as np

from f1tenth_gym_jax import make


def _load_example_module(name: str):
    path = pathlib.Path(__file__).parent.parent / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExamples(unittest.TestCase):
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
