Installation
============

``f1tenth_gym_jax`` is a Python package for a JAX-compatible F1TENTH racing
environment.

Using uv
--------

``uv`` is the supported install path for this repository. It reads
``uv.lock`` and the git source configured for ``jax-pf``.

.. code:: bash

    git clone https://github.com/f1tenth/f1tenth_gym_jax.git
    cd f1tenth_gym_jax
    uv sync

Install optional dependencies for specific workflows:

.. code:: bash

    uv sync --extra examples  # plotting, video, and track generation examples
    uv sync --extra rl        # PPO training/evaluation dependencies
    uv sync --extra docs      # Sphinx documentation build
    uv sync --extra cuda      # JAX CUDA 12 support

Using pip
---------

Plain ``pip install -e .`` does not read ``[tool.uv.sources]`` from
``pyproject.toml``. If you need a pip-only workflow, install the git-backed
``jax-pf`` dependency first, then install this package.

.. code:: bash

    git clone https://github.com/f1tenth/f1tenth_gym_jax.git
    cd f1tenth_gym_jax
    python -m pip install "jax-pf @ git+https://github.com/hzheng40/jax_pf"
    python -m pip install -e .

Docker
------

The default Docker image installs the standard dependency set and uses
offscreen Qt rendering, matching the headless CI smoke tests.

.. code:: bash

    docker build -t f1tenth_gym_jax -f Dockerfile .
    docker run -it f1tenth_gym_jax

Map cache
---------

Bundled maps are loaded from the installed package. Downloaded maps are stored
under ``$XDG_CACHE_HOME/f1tenth_gym_jax/maps`` by default. Set
``F1TENTH_GYM_JAX_MAP_DIR`` to use a different writable map cache.
