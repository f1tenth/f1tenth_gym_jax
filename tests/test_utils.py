import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.utils import LogWrapper, batchify, unbatchify


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
