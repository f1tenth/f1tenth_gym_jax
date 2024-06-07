import pathlib
import time
import unittest

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from f1tenth_gym.envs.track import Track, cubic_spline

class TestFrenet(unittest.TestCase):
    def test_calc_curvature(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSpline2D(circle_x, circle_y)
        # Test the curvature at the four corners of the circle
        # The curvature of a circle is 1/radius
        self.assertAlmostEqual(track.calc_curvature(0), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(np.pi / 2), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(np.pi), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(3 * np.pi / 2), 1, places=3)

if __name__ == "__main__":
    TestFrenet().test_calc_curvature()