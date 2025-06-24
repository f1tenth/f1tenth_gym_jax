# MIT License
import time
import unittest

import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from f1tenth_gym_jax.envs.collision_models import get_vertices, collision, collision_map
from f1tenth_gym_jax import make

class CollisionTests(unittest.TestCase):
    def setUp(self):
        # set up track
        self.env = make("Spielberg_3_noscan_time_v0")

        # car size
        self.length = 0.32
        self.width = 0.22

        # test poses
        self.test_pose = jnp.array([2.3, 6.7, 0.8])
        self.all_poses = jnp.tile(self.test_pose, (1000, 1))
        rng = jax.random.key(0)
        noises = jax.random.normal(rng, shape=(1000, 3))/10
        self.all_poses_noised = self.all_poses + noises

        # test poses for map collision check
        # grid of perturbations around the test pose in both x and y
        perturbations = jnp.linspace(-2.5, 2.5, 100)
        self.map_test_poses = jnp.stack(jnp.meshgrid(perturbations, perturbations), axis=-1).reshape(-1, 2)
        self.map_test_poses = jnp.hstack((self.map_test_poses, jnp.zeros((self.map_test_poses.shape[0], 1))))
        self.map_test_vertices = jax.vmap(
            partial(get_vertices, length=self.length, width=self.width), in_axes=[0]
        )(self.map_test_poses)


        # Collision check body
        self.all_vertices = jax.vmap(
            partial(get_vertices, length=self.length, width=self.width), in_axes=[0]
        )(self.all_poses_noised)
        # pairwise vertices
        pi1, pi2 = jnp.triu_indices(self.all_vertices.shape[0], 1)
        self.pairwise_vertices = jnp.concatenate(
            (self.all_vertices[pi1], self.all_vertices[pi2]), axis=-1
        )

    def test_get_vert(self, debug=False):
        vertices = get_vertices(self.test_pose, self.length, self.width)
        rect = np.vstack((vertices, vertices[0, :]))
        if debug:
            import matplotlib.pyplot as plt

            plt.scatter(self.test_pose[0], self.test_pose[1], c="red")
            plt.plot(rect[:, 0], rect[:, 1])
            plt.xlim([1, 4])
            plt.ylim([5, 8])
            plt.axes().set_aspect("equal")
            plt.show()
        self.assertTrue(vertices.shape == (4, 2))

    def test_get_vert_fps(self):
        start = time.time()
        for _ in range(1000):
            get_vertices(self.test_pose, self.length, self.width)
        elapsed = time.time() - start
        fps = 1000 / elapsed
        print("get vertices fps:", fps)
        self.assertGreater(fps, 500)

    def test_random_collision(self):
        col = jax.vmap(collision, in_axes=[0])(self.pairwise_vertices)
        self.assertTrue(jnp.any(col))
        print(f"collisions: {jnp.sum(col)} / {col.shape[0]}")

    def test_fps(self):
        # also perturb the body but mainly want to test GJK speed
        start = time.time()
        for _ in range(1000):
            col = jax.vmap(collision, in_axes=[0])(self.pairwise_vertices)
        elapsed = time.time() - start
        fps = 1000 / elapsed
        print("sat fps:", fps)
        # self.assertGreater(fps, 500)  This is a platform dependent test, not ideal.

    def test_map_collision(self):
        col_map = collision_map(self.map_test_vertices, self.env.pixel_centers)
        print(f"map collisions: {jnp.sum(col_map)} / {col_map.shape[0]}")
        import matplotlib.pyplot as plt
        plt.plot(self.map_test_vertices[:, :, 0], self.map_test_vertices[:, :, 1], "o", markersize=1, alpha=0.5)
        plt.scatter(self.env.pixel_centers[:, 0], self.env.pixel_centers[:, 1], s=1)
        plt.show()
        self.assertTrue(jnp.any(col_map))

    
    def test_sat(self):
        # Test known collision case
        pose1 = jnp.array([0.0, 0.0, 0.6])
        pose2 = jnp.array([0.1, 0.0, 0.0])
        vert1 = get_vertices(pose1, self.length, self.width)
        vert2 = get_vertices(pose2, self.length, self.width)
        pair_vert = jnp.concatenate((vert1, vert2), axis=-1)[jnp.newaxis, ...]
        col = jax.vmap(collision, in_axes=[0])(pair_vert)
        self.assertTrue(col[0])

        # Test known non-collision case
        pose3 = jnp.array([1.0, 1.0, 0.0])
        vert3 = get_vertices(pose3, self.length, self.width)
        pair_vert2 = jnp.concatenate((vert1, vert3), axis=-1)[jnp.newaxis, ...]
        col2 = jax.vmap(collision, in_axes=[0])(pair_vert2)
        self.assertFalse(col2[0])