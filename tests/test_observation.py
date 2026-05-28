import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make


class TestObservationInterface(unittest.TestCase):
    def test_observation_layout_without_scan(self):
        env = make(
            "Spielberg_2_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        obs, _ = env.reset(jax.random.key(0))

        self.assertEqual(env.observation_space_ind["dynamics_state"], list(range(7)))
        self.assertEqual(
            env.observation_space_ind["other_agent_dynamics_state"],
            list(range(7, 11)),
        )
        self.assertEqual(env.observation_space_ind["scan"], [])
        self.assertEqual(obs["agent_0"].shape, (11,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(obs["agent_0"]))))

    def test_relative_agent_observation_uses_cartesian_frame(self):
        env = make(
            "Spielberg_2_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        obs, state = env.reset(jax.random.key(0))

        agent_pose = state.cartesian_states[0, [0, 1, 4]]
        other_pose = state.cartesian_states[1, [0, 1, 4]]
        relative_pose = other_pose - agent_pose
        expected = jnp.array(
            [
                relative_pose[0],
                relative_pose[1],
                state.cartesian_states[1, 3],
                jnp.arctan2(jnp.sin(relative_pose[2]), jnp.cos(relative_pose[2])),
            ]
        )

        np.testing.assert_allclose(obs["agent_0"][7:11], expected, rtol=1e-6)

    def test_observation_layout_without_other_agents(self):
        env = make(
            "Spielberg_2_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0",
            observe_others=False,
        )
        obs, _ = env.reset(jax.random.key(0))

        self.assertEqual(
            env.observation_space_ind["other_agent_dynamics_state"],
            [],
        )
        self.assertEqual(obs["agent_0"].shape, (7,))

    def test_kinematic_model_observation_layout(self):
        env = make(
            "Spielberg_2_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0",
            model="ks",
            observe_others=False,
        )
        obs, state = env.reset(jax.random.key(0))

        self.assertEqual(state.cartesian_states.shape, (2, 5))
        self.assertEqual(env.observation_space_ind["dynamics_state"], list(range(5)))
        self.assertEqual(
            env.observation_space_ind["other_agent_dynamics_state"],
            [],
        )
        self.assertEqual(obs["agent_0"].shape, (5,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(obs["agent_0"]))))

    def test_scan_observation_layout(self):
        env = make(
            "Spielberg_1_scan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        obs, state = env.reset(jax.random.key(0))

        scan_indices = env.observation_space_ind["scan"]
        scan_slice = slice(scan_indices[0], scan_indices[-1] + 1)
        self.assertEqual(len(scan_indices), env.params.num_beams)
        self.assertEqual(state.scans.shape, (1, env.params.num_beams))
        self.assertEqual(obs["agent_0"].shape, (7 + env.params.num_beams,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(obs["agent_0"][scan_slice]))))
