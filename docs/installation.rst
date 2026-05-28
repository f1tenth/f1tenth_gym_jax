Installation
============

``f1tenth_gym_jax`` is a Python package for a JAX-compatible F1TENTH racing
environment.

Using uv
--------

``uv`` is the official install path for this repository. Use Python 3.11, 3.12,
or 3.13. ``uv`` reads ``uv.lock``, the git source configured for ``jax-pf``,
and the dependency overrides required to keep the JAX stack consistent.

.. code:: bash

    git clone https://github.com/f1tenth/f1tenth_gym_jax.git
    cd f1tenth_gym_jax
    uv sync

Install optional dependencies for specific workflows:

.. code:: bash

    uv sync --extra examples  # plotting, web dashboard, and track generation examples
    uv sync --extra rl        # PPO training/evaluation dependencies
    uv sync --extra docs      # Sphinx documentation build
    uv sync --extra cuda      # JAX CUDA 13 support

The default install uses CPU JAX wheels. The ``cuda`` extra follows JAX's
``jax[cuda13]`` packaging and requires a compatible Linux NVIDIA driver.

Documentation
-------------

The documentation is built with Sphinx from the ``docs/`` directory. The
repository also includes a root ``.readthedocs.yaml`` so Read the Docs can build
the same Sphinx project with ``uv``.

.. code:: bash

    uv sync --extra docs
    uv run sphinx-build -W -b html docs docs/_build/html

Docker
------

The default Docker image installs the standard dependency set. Rollout
visualization is generated as a standalone HTML dashboard that can be opened in
any browser.

.. code:: bash

    docker build -t f1tenth_gym_jax -f Dockerfile .
    docker run -it f1tenth_gym_jax

Map cache
---------

Bundled maps are loaded from the installed package. Downloaded maps are stored
under ``$XDG_CACHE_HOME/f1tenth_gym_jax/maps`` by default. Set
``F1TENTH_GYM_JAX_MAP_DIR`` to use a different writable map cache.
