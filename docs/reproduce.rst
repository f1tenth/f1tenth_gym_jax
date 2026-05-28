.. _reproduce:

Reproducing experiments when using the environment
====================================================

Use ``uv sync --frozen`` to install the base versions recorded in ``uv.lock``.
For PPO training and evaluation, install the example and RL extras as well:

.. code:: bash

    uv sync --frozen --extra examples --extra rl

The examples create all simulation randomness from explicit JAX PRNG keys, so
reruns are deterministic when the same seed, environment ID, and dependency
versions are used.

The PPO examples save and load model parameters from ``trained_models_ppo/``.
For evaluation, keep the ``TrainConfig.env_name`` and ``TrainConfig.run_name``
values aligned with the model file names in that directory.
