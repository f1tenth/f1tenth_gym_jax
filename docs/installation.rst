Installation
============

``f1tenth_gym_jax`` is a Python package for a JAX-compatible F1TENTH racing
environment.

Using uv
--------

.. code:: bash

    git clone https://github.com/f1tenth/f1tenth_gym_jax.git
    cd f1tenth_gym_jax
    uv sync

Install optional dependencies for specific workflows:

.. code:: bash

    uv sync --extra examples  # plotting, video, and track generation examples
    uv sync --extra rl        # PPO training/evaluation dependencies
    uv sync --extra docs      # Sphinx documentation build

Using pip
---------

.. code:: bash

    git clone https://github.com/f1tenth/f1tenth_gym_jax.git
    cd f1tenth_gym_jax
    python -m pip install -e .

Docker
------

.. code:: bash

    docker build -t f1tenth_gym_jax -f Dockerfile .
    docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_jax
