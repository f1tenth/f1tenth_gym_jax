import pathlib
import tomllib
import unittest

import f1tenth_gym_jax


class TestMetadata(unittest.TestCase):
    def test_package_version_matches_project_metadata(self):
        pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        metadata = tomllib.loads(pyproject.read_text())["project"]

        self.assertEqual(f1tenth_gym_jax.__version__, metadata["version"])
