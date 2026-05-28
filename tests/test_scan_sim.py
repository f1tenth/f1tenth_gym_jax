import unittest

import jax
import jax.numpy as jnp

from f1tenth_gym_jax import make


class ScanTests(unittest.TestCase):
    def test_reset_generates_finite_scans(self):
        env = make(
            "Spielberg_1_scan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        obs, state = env.reset(jax.random.key(0))

        self.assertEqual(state.scans.shape, (1, env.params.num_beams))
        self.assertTrue(bool(jnp.all(jnp.isfinite(state.scans))))
        self.assertTrue(bool(jnp.all(state.scans < env.params.max_range + 1.0)))
        self.assertEqual(len(env.observation_space_ind["scan"]), env.params.num_beams)
        scan_indices = env.observation_space_ind["scan"]
        scan_slice = slice(scan_indices[0], scan_indices[-1] + 1)
        self.assertTrue(bool(jnp.allclose(obs["agent_0"][scan_slice], state.scans[0])))

    def test_step_refreshes_scans(self):
        env = make(
            "Spielberg_1_scan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        _, state = env.reset(jax.random.key(0))
        actions = {"agent_0": jnp.array([0.0, 0.2])}
        obs, state, _, _, _ = env.step(jax.random.key(1), state, actions)

        self.assertEqual(state.scans.shape, (1, env.params.num_beams))
        self.assertTrue(bool(jnp.all(jnp.isfinite(state.scans))))
        scan_indices = env.observation_space_ind["scan"]
        scan_slice = slice(scan_indices[0], scan_indices[-1] + 1)
        self.assertTrue(bool(jnp.allclose(obs["agent_0"][scan_slice], state.scans[0])))
