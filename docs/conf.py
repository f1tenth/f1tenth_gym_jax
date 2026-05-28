import os
import tomllib
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_METADATA = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())[
    "project"
]

source_suffix = ".rst"
source_encoding = "utf-8-sig"

# -- Theme -------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    # 'typekit_id': 'hiw1hhg',
    # 'analytics_id': '',
    # 'sticky_navigation': True  # Set to False to disable the sticky nav while scrolling.
    "logo_only": False,  # if we have a html_logo below, this shows /only/ the logo with no title text
    "collapse_navigation": False,  # Collapse navigation (False makes it tree-like)
    "prev_next_buttons_location": "bottom",
    # 'display_version': True,  # Display the docs version
    # 'navigation_depth': 4,  # Depth of the headers shown in the navigation bar
}
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "f1tenth",  # Username
    "github_repo": "f1tenth_gym_jax",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

html_favicon = "assets/f1_stickers_02.png"

html_logo = "assets/f1tenth_gym.svg"

# -- Project information -----------------------------------------------------

project = PROJECT_METADATA["name"]
copyright = "2021, Hongrui Zheng, Matthew O'Kelly, Aman Sinha"
author = "Hongrui Zheng"

# The full version, including alpha/beta/rc tags
release = PROJECT_METADATA["version"]
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
