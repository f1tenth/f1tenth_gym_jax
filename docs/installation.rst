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
