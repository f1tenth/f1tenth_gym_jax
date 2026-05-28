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

    def test_documentation_links_target_readthedocs(self):
        repo_root = pathlib.Path(__file__).parent.parent
        metadata = tomllib.loads((repo_root / "pyproject.toml").read_text())["project"]
        readme = (repo_root / "README.md").read_text()

        self.assertEqual(
            metadata["urls"]["Documentation"],
            "https://f1tenth-gym-jax.readthedocs.io/",
        )
        self.assertIn("readthedocs.org/projects/f1tenth-gym-jax/badge", readme)
        self.assertIn("f1tenth-gym-jax.readthedocs.io/en/latest", readme)

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

    def test_dev_dependencies_enable_notebook_formatting(self):
        pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        dev_dependencies = tomllib.loads(pyproject.read_text())["dependency-groups"][
            "dev"
        ]

        self.assertTrue(
            any(
                dependency.startswith("black[jupyter]")
                for dependency in dev_dependencies
            )
        )

    def test_readthedocs_uses_sphinx_with_uv_docs_extra(self):
        repo_root = pathlib.Path(__file__).parent.parent
        config = yaml.safe_load((repo_root / ".readthedocs.yaml").read_text())

        self.assertEqual(config["version"], 2)
        self.assertEqual(config["build"]["os"], "ubuntu-24.04")
        self.assertIn("python", config["build"]["tools"])
        self.assertEqual(config["sphinx"]["configuration"], "docs/conf.py")
        self.assertTrue(config["sphinx"]["fail_on_warning"])

        install = config["python"]["install"]
        self.assertEqual(len(install), 1)
        self.assertEqual(install[0]["method"], "uv")
        self.assertEqual(install[0]["command"], "sync")
        self.assertIn("docs", install[0]["extras"])

    def test_qt_rendering_dependencies_are_removed(self):
        repo_root = pathlib.Path(__file__).parent.parent
        metadata = tomllib.loads((repo_root / "pyproject.toml").read_text())
        dependency_text = "\n".join(
            metadata["project"]["dependencies"]
            + metadata["project"].get("optional-dependencies", {}).get("examples", [])
            + metadata["dependency-groups"]["dev"]
        ).lower()
        lock_text = (repo_root / "uv.lock").read_text().lower()

        for dependency in ("pyqt", "pyqtgraph", "pyopengl"):
            with self.subTest(dependency=dependency):
                self.assertNotIn(dependency, dependency_text)
                self.assertNotIn(f'name = "{dependency}', lock_text)

    def test_docker_context_excludes_local_artifacts(self):
        repo_root = pathlib.Path(__file__).parent.parent
        dockerfile = (repo_root / "Dockerfile").read_text()
        dockerignore = repo_root / ".dockerignore"
        patterns = {
            line.strip()
            for line in dockerignore.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertIn("COPY . /f1tenth_gym_jax", dockerfile)
        self.assertNotIn("f1tenth_gym_jax_rollout.gif", patterns)
        for pattern in (
            ".git/",
            ".venv/",
            "wandb/",
            "**/__pycache__/",
            ".pytest_cache/",
            "blah/",
            "build/",
            "dist/",
            "f1tenth_gym_jax_rollout.html",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, patterns)

    def test_generated_dashboard_artifact_is_ignored(self):
        repo_root = pathlib.Path(__file__).parent.parent
        gitignore = repo_root / ".gitignore"
        patterns = {
            line.strip()
            for line in gitignore.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertIn("f1tenth_gym_jax_rollout.html", patterns)
        self.assertNotIn("f1tenth_gym_jax_rollout.gif", patterns)

    def test_gitignore_has_no_duplicate_patterns(self):
        repo_root = pathlib.Path(__file__).parent.parent
        gitignore = repo_root / ".gitignore"
        patterns = [
            line.strip()
            for line in gitignore.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

        duplicates = sorted(
            pattern for pattern in set(patterns) if patterns.count(pattern) > 1
        )
        self.assertEqual(duplicates, [])
