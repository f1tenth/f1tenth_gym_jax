# MIT License
import time
import unittest

import numpy as np
from f1tenth_gym_jax.envs.collision_models import get_vertices, collision

class CollisionTests(unittest.TestCase):
    def setUp(self):
        # test params
        np.random.seed(1234)

        # Collision check body
        self.vertices1 = np.asarray([[4, 11.0], [5, 5], [9, 9], [10, 10]])

        # car size
        self.length = 0.32
        self.width = 0.22

    def test_get_vert(self, debug=False):
        test_pose = np.array([2.3, 6.7, 0.8])
        vertices = get_vertices(test_pose, self.length, self.width)
        rect = np.vstack((vertices, vertices[0, :]))
        if debug:
            import matplotlib.pyplot as plt

            plt.scatter(test_pose[0], test_pose[1], c="red")
            plt.plot(rect[:, 0], rect[:, 1])
            plt.xlim([1, 4])
            plt.ylim([5, 8])
            plt.axes().set_aspect("equal")
            plt.show()
        self.assertTrue(vertices.shape == (4, 2))

    def test_get_vert_fps(self):
        test_pose = np.array([2.3, 6.7, 0.8])
        start = time.time()
        for _ in range(1000):
            get_vertices(test_pose, self.length, self.width)
        elapsed = time.time() - start
        fps = 1000 / elapsed
        print("get vertices fps:", fps)
        self.assertGreater(fps, 500)

    def test_random_collision(self):
        # perturb the body by a small amount and make sure it all collides with the original body
        for _ in range(1000):
            a = self.vertices1 + np.random.normal(size=(self.vertices1.shape)) / 100.0
            b = self.vertices1 + np.random.normal(size=(self.vertices1.shape)) / 100.0
            self.assertTrue(collision(a, b))

    def test_fps(self):
        # also perturb the body but mainly want to test GJK speed
        start = time.time()
        for _ in range(1000):
            a = self.vertices1 + np.random.normal(size=(self.vertices1.shape)) / 100.0
            b = self.vertices1 + np.random.normal(size=(self.vertices1.shape)) / 100.0
            collision(a, b)
        elapsed = time.time() - start
        fps = 1000 / elapsed
        print("gjk fps:", fps)
        # self.assertGreater(fps, 500)  This is a platform dependent test, not ideal.
