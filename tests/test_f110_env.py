import unittest

import jax
import jax.numpy as jnp

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.f110_env import F110Env
from f1tenth_gym_jax.envs.utils import Param

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

    def test_agent_classes_reports_homogeneous_cars(self):
        env = make(BASE_ENV_ID)

        self.assertEqual(env.agent_classes, {"car": ["agent_0", "agent_1"]})

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

    def test_continuous_actions_are_always_available(self):
        env = make(BASE_ENV_ID)
        _, state = env.reset(jax.random.key(0))

        available_actions = env.get_avail_actions(state)

        self.assertEqual(set(available_actions), set(env.agents))
        for agent, mask in available_actions.items():
            with self.subTest(agent=agent):
                self.assertEqual(mask.shape, env.action_space(agent).shape)
                self.assertEqual(mask.dtype, jnp.dtype(bool))
                self.assertTrue(bool(jnp.all(mask)))

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

    def test_time_reward_penalizes_elapsed_control_time(self):
        env = make(
            "Spielberg_1_noscan_nocollision_time_acceleration+steeringvelocity_1_5_v0"
        )
        _, state = env.reset(jax.random.key(0))
        actions = {"agent_0": jnp.zeros((2,))}

        _, _, rewards, _, _ = env.step(jax.random.key(1), state, actions)

        expected_reward = -env.params.timestep * env.params.timestep_ratio
        self.assertAlmostEqual(float(rewards["agent_0"]), expected_reward)

    def test_smooth_single_track_model_steps_from_rest(self):
        env = make(
            "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0",
            model="st_smooth",
        )
        _, state = env.reset(jax.random.key(0))
        actions = {"agent_0": jnp.array([0.0, 1.0])}

        _, next_state, rewards, _, _ = env.step(jax.random.key(1), state, actions)

        self.assertTrue(bool(jnp.all(jnp.isfinite(next_state.cartesian_states))))
        self.assertTrue(bool(jnp.isfinite(rewards["agent_0"])))

    def test_invalid_reward_override_is_rejected(self):
        with self.assertRaises(ValueError):
            make(
                "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0",
                reward_type="notprogress",
            )

    def test_invalid_constructor_params_are_rejected(self):
        invalid_params = [
            ("integrator", "verlet", "integrator"),
            ("model", "bicycle", "dynamics model"),
            ("longitudinal_action_type", "throttle", "longitudinal action type"),
            ("steering_action_type", "wheel", "steering action type"),
            ("mu", 0.0, "surface friction coefficient"),
            ("C_Sf", 0.0, "front cornering stiffness"),
            ("C_Sr", 0.0, "rear cornering stiffness"),
            ("lf", 0.0, "front axle distance"),
            ("lr", 0.0, "rear axle distance"),
            ("h", 0.0, "center of gravity height"),
            ("m", 0.0, "vehicle mass"),
            ("I", 0.0, "vehicle inertia"),
            ("v_switch", 0.0, "switching velocity"),
            ("a_max", 0.0, "maximum acceleration"),
            ("width", 0.0, "vehicle width"),
            ("length", 0.0, "vehicle length"),
            ("timestep", 0.0, "timestep"),
            ("timestep_ratio", 0, "timestep ratio"),
            ("max_steps", 0, "max steps"),
            ("max_num_laps", 0, "max number of laps"),
            ("theta_dis", 0, "theta discretization"),
            ("num_beams", 1, "number of scan beams"),
            ("fov", 0.0, "field of view"),
            ("eps", 0.0, "scan epsilon"),
            ("max_range", 0.0, "max scan range"),
        ]

        for field, value, message in invalid_params:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    F110Env(params=Param(**{field: value}))

        with self.assertRaisesRegex(ValueError, "number of agents"):
            F110Env(num_agents=0)

    def test_constructor_rejects_ignored_parameter_overrides(self):
        with self.assertRaisesRegex(TypeError, "Use f1tenth_gym_jax.make"):
            F110Env(timestep=0.02)

    def test_invalid_constructor_bounds_are_rejected(self):
        invalid_bounds = [
            ({"s_min": 0.5, "s_max": 0.5}, "steering angle"),
            ({"sv_min": 1.0, "sv_max": -1.0}, "steering velocity"),
            ({"v_min": 3.0, "v_max": 3.0}, "velocity"),
        ]

        for kwargs, message in invalid_bounds:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    F110Env(params=Param(**kwargs))
