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
        self.assertEqual(metadata["requires-python"], ">=3.11,<3.14")

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

    def test_api_docs_include_all_source_modules(self):
        repo_root = pathlib.Path(__file__).parent.parent
        package_root = repo_root / "f1tenth_gym_jax"
        docs_text = "\n".join(
            path.read_text() for path in sorted((repo_root / "docs/api").glob("*.rst"))
        )

        modules = []
        for path in sorted(package_root.rglob("*.py")):
            if path.name == "__init__.py":
                continue

            module_path = path.relative_to(repo_root).with_suffix("")
            modules.append(".".join(module_path.parts))

        missing_modules = [
            module for module in modules if f".. automodule:: {module}" not in docs_text
        ]

        self.assertEqual(missing_modules, [])

    def test_sphinx_uses_numpy_style_docstrings(self):
        repo_root = pathlib.Path(__file__).parent.parent
        conf_text = (repo_root / "docs" / "conf.py").read_text()
        package_text = "\n".join(
            path.read_text()
            for path in sorted((repo_root / "f1tenth_gym_jax").rglob("*.py"))
        )

        self.assertIn("napoleon_numpy_docstring = True", conf_text)
        self.assertIn("napoleon_google_docstring = False", conf_text)
        for fragment in ("Args:", "Returns:", "return:"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, package_text)

    def test_ci_jobs_install_extras_used_by_their_commands(self):
        workflow_path = (
            pathlib.Path(__file__).parent.parent / ".github/workflows/ci.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text())

        def install_command(job_name):
            steps = workflow["jobs"][job_name]["steps"]
            return next(
                step for step in steps if step.get("name") == "Install dependencies"
            )["run"]

        for job_name in ("tests", "examples"):
            with self.subTest(job=job_name):
                command = install_command(job_name)
                self.assertIn("--extra examples", command)
                self.assertIn("--extra rl", command)

        docs_command = install_command("docs")
        self.assertIn("--extra docs", docs_command)
        self.assertNotIn("--dev", docs_command)

    def test_ci_package_job_checks_all_tracked_map_files(self):
        workflow_path = (
            pathlib.Path(__file__).parent.parent / ".github/workflows/ci.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text())
        package_steps = workflow["jobs"]["package"]["steps"]
        check_step = next(
            step for step in package_steps if step.get("name") == "Check packaged maps"
        )
        command = check_step["run"]

        self.assertIn('["git", "ls-files", "maps"]', command)
        self.assertIn('not Path(path).name.startswith(".")', command)
        self.assertNotIn('"maps/Spielberg/Spielberg.yaml"', command)

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

    def test_plotting_dependencies_belong_to_examples_extra(self):
        pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        metadata = tomllib.loads(pyproject.read_text())
        examples_dependencies = metadata["project"]["optional-dependencies"]["examples"]
        dev_dependencies = metadata["dependency-groups"]["dev"]

        self.assertTrue(
            any(
                dependency.startswith("matplotlib")
                for dependency in examples_dependencies
            )
        )
        self.assertFalse(
            any(dependency.startswith("matplotlib") for dependency in dev_dependencies)
        )

    def test_rl_extra_uses_jax7_compatible_distrax(self):
        pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        metadata = tomllib.loads(pyproject.read_text())
        rl_dependencies = metadata["project"]["optional-dependencies"]["rl"]

        self.assertIn("distrax>=0.1.7", rl_dependencies)

    def test_cuda_extra_uses_jax_cuda13(self):
        repo_root = pathlib.Path(__file__).parent.parent
        metadata = tomllib.loads((repo_root / "pyproject.toml").read_text())
        cuda_dependencies = metadata["project"]["optional-dependencies"]["cuda"]
        uv_overrides = metadata["tool"]["uv"]["override-dependencies"]
        lock_text = (repo_root / "uv.lock").read_text()
        docs_text = "\n".join(
            [
                (repo_root / "README.md").read_text(),
                (repo_root / "docs" / "installation.rst").read_text(),
            ]
        )

        self.assertTrue(
            any(
                dependency.startswith("jax[cuda13]") for dependency in cuda_dependencies
            )
        )
        self.assertTrue(
            any(
                dependency.startswith("jax-cuda13-plugin[with-cuda]")
                for dependency in cuda_dependencies
            )
        )
        self.assertIn("jax>=0.7.2,<0.8", metadata["project"]["dependencies"])
        self.assertIn("jax>=0.7.2,<0.8", uv_overrides)
        self.assertIn('name = "jax-cuda13-plugin"', lock_text)
        self.assertNotIn("jax-cuda12", lock_text)
        self.assertNotIn("jax[cuda12]", "\n".join(cuda_dependencies))
        self.assertIn("CUDA 13", docs_text)
        self.assertNotIn("CUDA 12", docs_text)

    def test_uv_is_only_documented_project_install_workflow(self):
        repo_root = pathlib.Path(__file__).parent.parent
        installation = (repo_root / "docs" / "installation.rst").read_text()
        readme = (repo_root / "README.md").read_text()
        install_docs = "\n".join([installation, readme])

        self.assertIn("official install path", install_docs)
        self.assertNotIn("Using pip", installation)
        self.assertNotIn("pip install -e .", install_docs)
        self.assertNotIn("python -m pip install -e", install_docs)

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
