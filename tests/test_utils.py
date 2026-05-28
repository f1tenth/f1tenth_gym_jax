import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.utils import (
    LogEnvState,
    LogWrapper,
    WorldStateWrapper,
    batchify,
    unbatchify,
)


class TestUtilities(unittest.TestCase):
    def test_batchify_and_unbatchify_roundtrip(self):
        agents = ["agent_0", "agent_1"]
        num_envs = 3
        values = {
            "agent_0": jnp.arange(6, dtype=jnp.float32).reshape(num_envs, 2),
            "agent_1": jnp.arange(6, 12, dtype=jnp.float32).reshape(num_envs, 2),
        }

        batched = batchify(values, agents, num_envs * len(agents))
        restored = unbatchify(batched, agents, num_envs, len(agents))

        self.assertEqual(batched.shape, (num_envs * len(agents), 2))
        np.testing.assert_allclose(restored["agent_0"], values["agent_0"])
        np.testing.assert_allclose(restored["agent_1"], values["agent_1"])

    def test_log_wrapper_adds_episode_info(self):
        env = LogWrapper(
            make(
                "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_2_v0"
            )
        )
        obs, state = env.reset(jax.random.key(0))
        action = {"agent_0": jnp.zeros((2,))}

        obs, state, reward, done, info = env.step(jax.random.key(1), state, action)

        self.assertEqual(set(obs), {"agent_0"})
        self.assertEqual(set(reward), {"agent_0"})
        self.assertIn("returned_episode_returns", info)
        self.assertIn("returned_episode_lengths", info)
        self.assertIn("returned_episode", info)
        self.assertEqual(info["returned_episode_returns"].shape, (1,))
        self.assertIsInstance(state, LogEnvState)

    def test_log_wrapper_resets_episode_counters_after_done(self):
        env = LogWrapper(
            make(
                "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_1_v0"
            )
        )
        _, state = env.reset(jax.random.key(0))
        action = {"agent_0": jnp.zeros((2,))}

        _, state, _, done, info = env.step(jax.random.key(1), state, action)

        self.assertTrue(bool(done["__all__"]))
        np.testing.assert_allclose(info["returned_episode_lengths"], np.array([1]))
        np.testing.assert_allclose(state.episode_lengths, np.array([0]))

    def test_world_state_wrapper_adds_flattened_joint_observation(self):
        env = WorldStateWrapper(
            make(
                "Spielberg_2_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
            )
        )

        obs, state = env.reset(jax.random.key(0))

        expected = jnp.concatenate([obs["agent_0"], obs["agent_1"]])
        self.assertEqual(set(obs), {"agent_0", "agent_1", "world_state"})
        self.assertEqual(obs["world_state"].shape, (2, expected.shape[0]))
        np.testing.assert_allclose(obs["world_state"][0], expected)
        np.testing.assert_allclose(obs["world_state"][1], expected)

        actions = {agent: jnp.zeros((2,)) for agent in env.agents}
        next_obs, _, _, _, _ = env.step(jax.random.key(1), state, actions)
        self.assertEqual(next_obs["world_state"].shape, (2, expected.shape[0]))
