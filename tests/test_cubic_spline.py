import unittest

import numpy as np

from f1tenth_gym_jax.envs.track import cubic_spline


class TestCubicSpline(unittest.TestCase):
    def _circle_track(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        return cubic_spline.CubicSplineND(circle_x, circle_y)

    def _assert_angles_close(self, actual, expected, places=2):
        diff = np.arctan2(np.sin(actual - expected), np.cos(actual - expected))
        self.assertAlmostEqual(diff, 0, places=places)

    def test_calc_curvature(self):
        track = self._circle_track()
        # Test the curvature at the four corners of the circle
        # The curvature of a circle is 1/radius
        self.assertAlmostEqual(track.calc_curvature(0), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(np.pi / 2), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(np.pi), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(3 * np.pi / 2), 1, places=3)

    def test_calc_yaw(self):
        track = self._circle_track()
        # Test the yaw at the four corners of the circle
        # The yaw of a circle is s + pi/2
        self._assert_angles_close(track.calc_yaw(0), np.pi / 2)
        self._assert_angles_close(track.calc_yaw(np.pi / 2), np.pi)
        self._assert_angles_close(track.calc_yaw(np.pi), 3 * np.pi / 2)
        self._assert_angles_close(track.calc_yaw(3 * np.pi / 2), 0)

    def test_calc_position(self):
        track = self._circle_track()
        # Test the position at the four corners of the circle
        # The position of a circle is (x, y) = (cos(s), sin(s))
        self.assertTrue(
            np.allclose(track.calc_position(0), np.array([1, 0]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(track.calc_position(np.pi / 2), np.array([0, 1]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(track.calc_position(np.pi), np.array([-1, 0]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(
                track.calc_position(3 * np.pi / 2), np.array([0, -1]), atol=1e-3
            )
        )

    def test_calc_position_uses_actual_spline_segments(self):
        track = cubic_spline.CubicSplineND(
            np.array([0.0, 0.2, 3.0, 3.5]),
            np.array([0.0, 2.0, 2.2, 0.0]),
        )

        for s in (2.2, 3.0, 5.0):
            with self.subTest(s=s):
                expected = np.asarray(track.spline(s)[:2])
                np.testing.assert_allclose(track.calc_position(s), expected)
                np.testing.assert_allclose(
                    np.asarray(track.calc_position_jax(s)),
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_calc_arclength(self):
        track = self._circle_track()
        # Test the arclength at the four corners of the circle
        self.assertAlmostEqual(track.calc_arclength(1, 0, 0)[0], 0, places=2)
        self.assertAlmostEqual(track.calc_arclength(0, 1, 0)[0], np.pi / 2, places=2)
        self.assertAlmostEqual(
            track.calc_arclength(-1, 0, np.pi / 2)[0], np.pi, places=2
        )
        self.assertAlmostEqual(
            track.calc_arclength(0, -1, np.pi)[0], 3 * np.pi / 2, places=2
        )

    def test_calc_arclength_slow(self):
        track = self._circle_track()
        # Test the arclength at the four corners of the circle
        self.assertAlmostEqual(track.calc_arclength_slow(1, 0)[0], 0, places=2)
        self.assertAlmostEqual(track.calc_arclength_slow(0, 1)[0], np.pi / 2, places=2)
        self.assertAlmostEqual(track.calc_arclength_slow(-1, 0)[0], np.pi, places=2)
        self.assertAlmostEqual(
            track.calc_arclength_slow(0, -1)[0], 3 * np.pi / 2, places=2
        )

    def test_jax_arclength_includes_closing_segment(self):
        track = cubic_spline.CubicSplineND(
            np.array([1.0, 0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.0, -1.0]),
        )

        point = (0.5, -0.5)
        s, ey = track.calc_arclength(*point)
        s_jax, ey_jax = track.calc_arclength_jax(*point)

        self.assertEqual(track.points_jax.shape, track.points.shape)
        self.assertAlmostEqual(float(s_jax), s, places=5)
        self.assertAlmostEqual(float(ey_jax), ey, places=5)
