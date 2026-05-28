.. _reproduce:

Reproducing experiments when using the environment
====================================================

Use ``uv sync --frozen`` to install the versions recorded in ``uv.lock``.
The examples create all simulation randomness from explicit JAX PRNG keys, so
reruns are deterministic when the same seed, environment ID, and dependency
versions are used.

The PPO examples save and load model parameters from ``trained_models_ppo/``.
For evaluation, keep the ``TrainConfig.env_name`` and ``TrainConfig.run_name``
values aligned with the model file names in that directory.
