import pathlib
import tomllib
import unittest

import yaml

import f1tenth_gym_jax


class TestMetadata(unittest.TestCase):
    def test_package_version_matches_project_metadata(self):
        pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        metadata = tomllib.loads(pyproject.read_text())["project"]

        self.assertEqual(f1tenth_gym_jax.__version__, metadata["version"])

    def test_ci_test_job_installs_extras_used_by_tested_examples(self):
        workflow_path = (
            pathlib.Path(__file__).parent.parent / ".github/workflows/ci.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text())
        test_steps = workflow["jobs"]["tests"]["steps"]
        install_step = next(
            step for step in test_steps if step.get("name") == "Install dependencies"
        )

        self.assertIn("--extra examples", install_step["run"])
        self.assertIn("--extra rl", install_step["run"])
