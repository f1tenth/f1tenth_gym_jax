import ast
import importlib.util
import json
import pathlib
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from f1tenth_gym_jax import make
from f1tenth_gym_jax.registration import _parse_scenario


def _load_example_module(name: str):
    path = pathlib.Path(__file__).parent.parent / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluate_train_config_default(node: ast.AST, values: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(str(_evaluate_train_config_default(value.value, values)))
            else:
                raise ValueError(f"Unsupported f-string value: {ast.dump(value)}")
        return "".join(parts)
    raise ValueError(f"Unsupported TrainConfig default: {ast.dump(node)}")


def _read_train_config_defaults() -> dict[str, object]:
    path = pathlib.Path(__file__).parent.parent / "examples" / "train_ppo_example.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    train_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrainConfig"
    )

    values = {}
    for node in train_config.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id not in {"num_agents", "env_name", "run_name"}:
            continue
        values[node.target.id] = _evaluate_train_config_default(node.value, values)

    return values


def _ppo_artifact_pairs() -> dict[tuple[str, str], set[str]]:
    artifact_dir = pathlib.Path(__file__).parent.parent / "trained_models_ppo"
    artifacts = {}
    for path in sorted(artifact_dir.glob("*/*_params.safetensors")):
        for kind in ("actor", "critic"):
            suffix = f"_{kind}_params.safetensors"
            if path.name.endswith(suffix):
                env_name = path.name.removesuffix(suffix)
                artifacts.setdefault((path.parent.name, env_name), set()).add(kind)
                break
        else:
            raise AssertionError(f"Unexpected PPO artifact name: {path}")
    return artifacts


_BANNED_STALE_EXAMPLE_FRAGMENTS = (
    "f110_gym",
    "gym.make",
    "gymnasium",
    "jax.random.PRNGKey",
)


class TestExamples(unittest.TestCase):
    def test_ppo_train_config_default_model_files_are_checked_in(self):
        defaults = _read_train_config_defaults()
        _parse_scenario(defaults["env_name"])

        model_stem = (
            pathlib.Path(__file__).parent.parent
            / "trained_models_ppo"
            / defaults["run_name"]
            / defaults["env_name"]
        )
        self.assertTrue(
            model_stem.with_name(
                f"{model_stem.name}_actor_params.safetensors"
            ).is_file()
        )
        self.assertTrue(
            model_stem.with_name(
                f"{model_stem.name}_critic_params.safetensors"
            ).is_file()
        )

    def test_ppo_train_config_derived_fields_are_normalized(self):
        train_ppo_example = _load_example_module("train_ppo_example")

        config = train_ppo_example.normalize_train_config(
            train_ppo_example.TrainConfig(
                num_envs=2,
                num_steps=4,
                total_timesteps=32,
                num_minibatches=1,
            )
        )

        self.assertEqual(config.num_actors, config.num_agents * config.num_envs)
        self.assertEqual(config.num_updates, 4)
        self.assertEqual(config.minibatch_size, 8)

    def test_ppo_train_config_rejects_invalid_rollout_shape(self):
        train_ppo_example = _load_example_module("train_ppo_example")

        invalid_configs = [
            (
                "total_timesteps",
                train_ppo_example.TrainConfig(
                    total_timesteps=1,
                    num_envs=2,
                    num_steps=2,
                    num_minibatches=1,
                ),
            ),
            (
                "num_minibatches",
                train_ppo_example.TrainConfig(num_envs=2, num_minibatches=3),
            ),
            (
                "num_envs",
                train_ppo_example.TrainConfig(num_envs=0),
            ),
        ]

        for field, config in invalid_configs:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    train_ppo_example.normalize_train_config(config)

    def test_ppo_trained_model_artifacts_are_paired_and_parseable(self):
        artifacts = _ppo_artifact_pairs()
        self.assertGreater(len(artifacts), 0)

        for (run_name, env_name), kinds in artifacts.items():
            with self.subTest(run_name=run_name, env_name=env_name):
                self.assertEqual(kinds, {"actor", "critic"})
                _parse_scenario(env_name)

    def test_notebook_examples_use_current_jax_environment_api(self):
        examples_dir = pathlib.Path(__file__).parent.parent / "examples"

        for notebook_name in ("benchmark_example.ipynb", "rendering_example.ipynb"):
            with self.subTest(notebook=notebook_name):
                notebook = json.loads((examples_dir / notebook_name).read_text())
                source = "".join(
                    line
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                    for line in cell["source"]
                )

                self.assertIn("from f1tenth_gym_jax import make", source)
                self.assertIn("jax.random.key", source)
                for fragment in _BANNED_STALE_EXAMPLE_FRAGMENTS:
                    self.assertNotIn(fragment, source)
                self.assertNotIn("[:, 100,", source)
                self.assertNotIn("reshape((num_agents, num_envs, -1))", source)

                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])

    def test_python_examples_use_current_jax_environment_api(self):
        examples_dir = pathlib.Path(__file__).parent.parent / "examples"
        environment_examples = {
            "eval_ppo_example.py",
            "mppi_example.py",
            "render_dashboard.py",
            "run_in_empty_track.py",
            "train_ppo_example.py",
            "waypoint_follow.py",
        }

        for example_name in environment_examples:
            with self.subTest(example=example_name):
                source = (examples_dir / example_name).read_text()

                self.assertIn("from f1tenth_gym_jax import make", source)
                self.assertIn("make(", source)
                for fragment in _BANNED_STALE_EXAMPLE_FRAGMENTS:
                    self.assertNotIn(fragment, source)

    def test_empty_track_example_runs_without_rendering(self):
        run_in_empty_track = _load_example_module("run_in_empty_track")

        trajectory = run_in_empty_track.rollout(num_steps=1, render=False)

        self.assertEqual(trajectory.shape, (1, 1, 1, 7))
        self.assertTrue(np.isfinite(trajectory).all())

    def test_empty_track_example_rejects_non_positive_steps(self):
        run_in_empty_track = _load_example_module("run_in_empty_track")

        with self.assertRaisesRegex(ValueError, "num_steps must be positive"):
            run_in_empty_track.rollout(num_steps=0, render=False)

    def test_waypoint_follow_example_runs_without_rendering(self):
        waypoint_follow = _load_example_module("waypoint_follow")

        _, all_runner_state = waypoint_follow.run_waypoint_follow(
            num_agents=1,
            num_envs=1,
            num_steps=1,
            render=False,
        )
        trajectory = np.asarray(all_runner_state[0].cartesian_states)

        self.assertEqual(trajectory.shape, (1, 1, 1, 7))
        self.assertTrue(np.isfinite(trajectory).all())

    def test_waypoint_follow_example_rejects_non_positive_counts(self):
        waypoint_follow = _load_example_module("waypoint_follow")

        invalid_kwargs = [
            ("num_agents", 0),
            ("num_envs", 0),
            ("num_steps", 0),
        ]

        for field, value in invalid_kwargs:
            with self.subTest(field=field):
                kwargs = {
                    "num_agents": 1,
                    "num_envs": 1,
                    "num_steps": 1,
                    field: value,
                    "render": False,
                }
                with self.assertRaisesRegex(ValueError, field):
                    waypoint_follow.run_waypoint_follow(**kwargs)

    def test_mppi_waypoint_geometry_uses_raceline_arclength_and_xy(self):
        mppi_example = _load_example_module("mppi_example")
        env = make(
            "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        config = mppi_example.MPPIConfig(n_samples=2, n_steps=2)

        mppi = mppi_example.MPPI(config, env, jax.random.key(0))

        expected_xy = np.column_stack((env.track.raceline.xs, env.track.raceline.ys))
        expected_distances = np.linalg.norm(expected_xy[1:] - expected_xy[:-1], axis=1)
        np.testing.assert_allclose(
            np.asarray(mppi.waypoints[:, 0]), env.track.raceline.s
        )
        np.testing.assert_allclose(
            np.asarray(mppi.waypoint_distances),
            expected_distances,
            rtol=1e-6,
            atol=1e-5,
        )

    def test_mppi_reference_starts_at_projected_segment_arclength(self):
        mppi_example = _load_example_module("mppi_example")
        env = make(
            "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_5_v0"
        )
        config = mppi_example.MPPIConfig(n_samples=2, n_steps=2)
        mppi = mppi_example.MPPI(config, env, jax.random.key(0))

        raceline = env.track.raceline
        midpoint = np.array(
            [
                0.5 * (raceline.xs[0] + raceline.xs[1]),
                0.5 * (raceline.ys[0] + raceline.ys[1]),
            ]
        )
        state = jnp.array(
            [midpoint[0], midpoint[1], 0.0, 1.0, raceline.psis[0], 0.0, 0.0]
        )

        reference, segment_index = mppi.get_ref(state, n_steps=1)

        expected_s = raceline.s[0] + 0.5 * (raceline.s[1] - raceline.s[0])
        self.assertEqual(int(segment_index), 0)
        self.assertAlmostEqual(float(reference[0, 4]), expected_s, places=5)

    def test_mppi_example_runs_without_plots_or_rendering(self):
        mppi_example = _load_example_module("mppi_example")
        config = mppi_example.MPPIConfig(n_samples=2, n_steps=2)

        _, all_runner_state, all_reward, all_done = mppi_example.run_mppi(
            num_agents=1,
            num_envs=1,
            num_steps=1,
            config=config,
            plot=False,
            render=False,
        )
        trajectory = np.asarray(all_runner_state[0].cartesian_states)

        self.assertEqual(trajectory.shape, (1, 1, 1, 7))
        self.assertEqual(np.asarray(all_reward).shape, (1, 1, 1))
        self.assertEqual(np.asarray(all_done).shape, (1, 1, 1))
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertTrue(np.isfinite(np.asarray(all_reward)).all())

    def test_mppi_example_rejects_non_positive_counts(self):
        mppi_example = _load_example_module("mppi_example")

        invalid_run_kwargs = [
            ("num_agents", 0),
            ("num_envs", 0),
            ("num_steps", 0),
        ]
        for field, value in invalid_run_kwargs:
            with self.subTest(field=field):
                kwargs = {
                    "num_agents": 1,
                    "num_envs": 1,
                    "num_steps": 1,
                    field: value,
                    "config": mppi_example.MPPIConfig(n_samples=2, n_steps=2),
                    "plot": False,
                    "render": False,
                }
                with self.assertRaisesRegex(ValueError, field):
                    mppi_example.run_mppi(**kwargs)

        invalid_configs = [
            ("config.n_samples", mppi_example.MPPIConfig(n_samples=0, n_steps=2)),
            ("config.n_steps", mppi_example.MPPIConfig(n_samples=2, n_steps=0)),
        ]
        for field, config in invalid_configs:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    mppi_example.run_mppi(
                        num_agents=1,
                        num_envs=1,
                        num_steps=1,
                        config=config,
                        plot=False,
                        render=False,
                    )

    def test_render_dashboard_writes_requested_html(self):
        render_dashboard = _load_example_module("render_dashboard")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "rollout.html"
            rendered = render_dashboard.save_dashboard(num_steps=1, output=output)

            self.assertEqual(rendered, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            self.assertIn("Timestep scrubber", output.read_text())

    def test_random_trackgen_uses_yaml_resolution_for_centerline_scale(self):
        random_trackgen = _load_example_module("random_trackgen")
        track = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
                [0.0, 0.0],
            ]
        )
        track_int = track + np.array([1.0, 1.0])
        track_ext = track - np.array([1.0, 1.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = pathlib.Path(tmpdir)
            random_trackgen.convert_track(
                track, track_int, track_ext, "RandomTrack0", outdir
            )

            track_dir = outdir / "RandomTrack0"
            with (track_dir / "RandomTrack0.yaml").open() as yaml_file:
                metadata = yaml.safe_load(yaml_file)
            centerline = np.loadtxt(
                track_dir / "RandomTrack0_centerline.csv",
                delimiter=",",
                comments="#",
            )

            self.assertEqual(metadata["resolution"], random_trackgen.MAP_RESOLUTION)
            self.assertTrue((track_dir / metadata["image"]).exists())
            np.testing.assert_allclose(centerline[0, :2], np.array([0.0, 0.0]))
            np.testing.assert_allclose(
                centerline[:, 2:],
                random_trackgen.TRACK_WIDTH * random_trackgen.MAP_RESOLUTION,
            )

    def test_random_trackgen_does_not_reference_stale_gym_url(self):
        source = (
            pathlib.Path(__file__).parent.parent / "examples" / "random_trackgen.py"
        ).read_text()

        self.assertNotIn("https://gym.openai.com", source)
