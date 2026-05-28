import unittest

import jax
import jax.numpy as jnp

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.f110_env import F110Env

BASE_ENV_ID = (
    "Spielberg_2_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
)


class TestF110Env(unittest.TestCase):
    def test_make_returns_jax_environment(self):
        env = make(BASE_ENV_ID)

        self.assertIsInstance(env, F110Env)
        self.assertEqual(env.num_agents, 2)
        self.assertEqual(env.agents, ["agent_0", "agent_1"])
        self.assertFalse(env.params.produce_scans)
        self.assertFalse(env.params.collision_on)
        self.assertEqual(env.params.reward_type, "progress")
        self.assertEqual(env.params.max_steps, 5)

    def test_reset_observation_and_state_shapes(self):
        env = make(BASE_ENV_ID)
        obs, state = env.reset(jax.random.key(0))

        self.assertEqual(set(obs), set(env.agents))
        self.assertEqual(state.cartesian_states.shape, (2, 7))
        self.assertEqual(state.frenet_states.shape, (2, 3))
        self.assertEqual(state.collisions.shape, (2,))
        self.assertEqual(obs["agent_0"].shape, env.observation_space("agent_0").shape)
        self.assertEqual(obs["agent_0"].shape, (11,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(obs["agent_0"]))))

    def test_step_uses_agent_action_dict(self):
        env = make(BASE_ENV_ID)
        obs, state = env.reset(jax.random.key(0))
        actions = {agent: jnp.array([0.0, 0.5]) for agent in env.agents}

        obs, next_state, rewards, dones, infos = env.step(
            jax.random.key(1), state, actions
        )

        self.assertEqual(set(obs), set(env.agents))
        self.assertEqual(set(rewards), set(env.agents))
        self.assertEqual(set(dones), set(env.agents + ["__all__"]))
        self.assertEqual(infos, {})
        self.assertEqual(
            next_state.cartesian_states.shape, state.cartesian_states.shape
        )
        self.assertFalse(bool(dones["__all__"]))

    def test_single_agent_collision_mode_steps(self):
        env = make(
            "Spielberg_1_noscan_collision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        obs, state = env.reset(jax.random.key(0))
        actions = {"agent_0": jnp.array([0.0, 0.5])}

        obs, next_state, rewards, dones, infos = env.step(
            jax.random.key(1), state, actions
        )

        self.assertEqual(set(obs), {"agent_0"})
        self.assertEqual(set(rewards), {"agent_0"})
        self.assertEqual(set(dones), {"agent_0", "__all__"})
        self.assertEqual(infos, {})
        self.assertEqual(next_state.cartesian_states.shape, (1, env.state_size))
        self.assertTrue(bool(jnp.isfinite(rewards["agent_0"])))

    def test_step_auto_resets_when_all_agents_done(self):
        env = make(BASE_ENV_ID)
        _, state = env.reset(jax.random.key(0))
        actions = {agent: jnp.zeros((2,)) for agent in env.agents}
        done = False

        for step in range(env.params.max_steps):
            _, state, _, dones, _ = env.step(jax.random.key(step + 1), state, actions)
            done = bool(dones["__all__"])

        self.assertTrue(done)
        self.assertEqual(int(state.step), 0)

    def test_action_space_matches_acceleration_steeringvelocity_controls(self):
        env = make(BASE_ENV_ID)
        action_space = env.action_space("agent_0")

        self.assertEqual(action_space.shape, (2,))
        self.assertTrue(
            bool(
                jnp.allclose(
                    action_space.low, jnp.array([env.params.sv_min, -env.params.a_max])
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    action_space.high, jnp.array([env.params.sv_max, env.params.a_max])
                )
            )
        )

        action = action_space.sample(jax.random.key(0))
        self.assertTrue(bool(action_space.contains(action)))

    def test_action_space_matches_velocity_steeringangle_controls(self):
        env = make(
            "Spielberg_1_noscan_nocollision_progress_velocity+steeringangle_1_5_v0"
        )
        action_space = env.action_space("agent_0")

        self.assertEqual(action_space.shape, (2,))
        self.assertTrue(
            bool(
                jnp.allclose(
                    action_space.low, jnp.array([env.params.s_min, env.params.v_min])
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    action_space.high, jnp.array([env.params.s_max, env.params.v_max])
                )
            )
        )

        action = action_space.sample(jax.random.key(1))
        self.assertTrue(bool(action_space.contains(action)))
