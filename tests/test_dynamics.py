import unittest

import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax.envs.dynamic_models import (
    accl_constraints,
    pid_accl,
    pid_steer,
    steering_constraint,
    upper_accel_limit,
    vehicle_dynamics_ks,
    vehicle_dynamics_st_smooth,
    vehicle_dynamics_st_switching,
)
from f1tenth_gym_jax.envs.utils import Param


class DynamicsTest(unittest.TestCase):
    def setUp(self):
        self.params = Param()

    def test_acceleration_constraints(self):
        self.assertAlmostEqual(float(upper_accel_limit(1.0, 10.0, 5.0)), 10.0)
        self.assertAlmostEqual(float(upper_accel_limit(10.0, 10.0, 5.0)), 5.0)
        self.assertAlmostEqual(
            float(accl_constraints(0.0, -1.0, 5.0, 10.0, 0.0, 20.0)), 0.0
        )
        self.assertAlmostEqual(
            float(accl_constraints(5.0, 20.0, 5.0, 10.0, 0.0, 20.0)), 10.0
        )
        self.assertAlmostEqual(
            float(accl_constraints(5.0, -20.0, 5.0, 10.0, 0.0, 20.0)), -10.0
        )

    def test_steering_constraints(self):
        self.assertAlmostEqual(
            float(steering_constraint(0.0, 10.0, -0.4, 0.4, -3.0, 3.0)), 3.0
        )
        self.assertAlmostEqual(
            float(steering_constraint(0.0, -10.0, -0.4, 0.4, -3.0, 3.0)), -3.0
        )
        self.assertAlmostEqual(
            float(steering_constraint(0.4, 1.0, -0.4, 0.4, -3.0, 3.0)), 0.0
        )
        self.assertAlmostEqual(
            float(steering_constraint(-0.4, -1.0, -0.4, 0.4, -3.0, 3.0)), 0.0
        )

    def test_kinematic_dynamics(self):
        x_and_u = jnp.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.5, 1.0])
        derivative = vehicle_dynamics_ks(x_and_u, self.params)
        np.testing.assert_allclose(
            np.asarray(derivative),
            np.array([2.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
            atol=1e-6,
        )

    def test_single_track_dynamics_are_finite(self):
        x_and_u = jnp.array([0.0, 0.0, 0.05, 3.0, 0.1, 0.0, 0.0, 0.2, 0.5])
        switching = vehicle_dynamics_st_switching(x_and_u, self.params)
        smooth = vehicle_dynamics_st_smooth(x_and_u, self.params)

        self.assertEqual(switching.shape, (9,))
        self.assertEqual(smooth.shape, (9,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(switching))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(smooth))))

    def test_pid_helpers(self):
        self.assertAlmostEqual(float(pid_steer(0.2, 0.0, 3.2)), 3.2)
        self.assertAlmostEqual(float(pid_steer(-0.2, 0.0, 3.2)), -3.2)
        self.assertAlmostEqual(float(pid_steer(0.0, 0.0, 3.2)), 0.0)
        self.assertGreater(float(pid_accl(5.0, 1.0, 10.0, 20.0, -5.0)), 0.0)
        self.assertLess(float(pid_accl(1.0, 5.0, 10.0, 20.0, -5.0)), 0.0)
