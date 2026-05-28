import unittest

import jax
import jax.numpy as jnp

from f1tenth_gym_jax.envs.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple


class TestSpaces(unittest.TestCase):
    def test_discrete_space(self):
        space = Discrete(3)

        self.assertEqual(space.shape, ())
        self.assertTrue(bool(space.contains(jnp.array(2))))
        self.assertFalse(bool(space.contains(jnp.array(3))))
        self.assertFalse(bool(space.contains(jnp.array([1]))))
        self.assertFalse(bool(space.contains(jnp.array(1.0))))
        with self.assertRaises(ValueError):
            Discrete(0)

    def test_multi_discrete_space(self):
        space = MultiDiscrete([2, 3])

        self.assertEqual(space.shape, (2,))
        self.assertTrue(bool(space.contains(jnp.array([1, 2]))))
        self.assertFalse(bool(space.contains(jnp.array([2, 2]))))
        self.assertFalse(bool(space.contains(jnp.array([1]))))
        self.assertFalse(bool(space.contains(jnp.array([1.0, 2.0]))))
        with self.assertRaises(ValueError):
            MultiDiscrete([2, 0])

    def test_box_space_supports_vector_bounds(self):
        space = Box(jnp.array([-1.0, 0.0]), jnp.array([1.0, 2.0]), (2,))

        sample = space.sample(jax.random.key(0))
        self.assertEqual(sample.shape, (2,))
        self.assertTrue(bool(space.contains(sample)))
        self.assertTrue(bool(space.contains(jnp.array([0.0, 1.0]))))
        self.assertFalse(bool(space.contains(jnp.array([2.0, 1.0]))))
        self.assertFalse(bool(space.contains(jnp.array([0.0]))))

    def test_box_space_samples_unbounded_dimensions(self):
        space = Box(
            jnp.array([-jnp.inf, 0.0, -jnp.inf, -1.0]),
            jnp.array([jnp.inf, jnp.inf, 1.0, 1.0]),
            (4,),
        )

        sample = space.sample(jax.random.key(1))

        self.assertEqual(sample.shape, (4,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(sample))))
        self.assertTrue(bool(space.contains(sample)))
        self.assertGreaterEqual(float(sample[1]), 0.0)
        self.assertLessEqual(float(sample[2]), 1.0)

    def test_box_space_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            Box(jnp.array([1.0]), jnp.array([0.0]), (1,))

    def test_dict_space_contains_dict_values(self):
        space = Dict({"a": Discrete(2), "b": Box(-1.0, 1.0, (2,))})

        self.assertTrue(
            bool(space.contains({"a": jnp.array(1), "b": jnp.array([0.0, 0.5])}))
        )
        self.assertFalse(
            bool(space.contains({"a": jnp.array(2), "b": jnp.array([0.0, 0.5])}))
        )
        self.assertFalse(bool(space.contains({"a": jnp.array(1)})))

    def test_tuple_space_contains_tuple_values(self):
        space = Tuple((Discrete(2), Box(-1.0, 1.0, (2,))))

        self.assertTrue(bool(space.contains((jnp.array(1), jnp.array([0.0, 0.5])))))
        self.assertFalse(bool(space.contains((jnp.array(1), jnp.array([2.0, 0.5])))))
        self.assertFalse(bool(space.contains((jnp.array(1),))))
