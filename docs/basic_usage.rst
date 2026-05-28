.. _basic_usage:

Basic Usage
===========

The current environment API is JAX-native. Create environments with
``f1tenth_gym_jax.make`` and step them with explicit PRNG keys, immutable state,
and per-agent action dictionaries.

.. code:: python

    import jax
    import jax.numpy as jnp

    from f1tenth_gym_jax import make

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_100_v0"
    )

    key = jax.random.key(0)
    obs, state = env.reset(key)

    actions = {"agent_0": jnp.array([0.0, 1.0])}
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, infos = env.step(step_key, state, actions)

``reset`` returns an observation dictionary and a JAX state object. ``step``
returns observations, the next state, per-agent rewards, per-agent done flags
plus ``"__all__"``, and an info dictionary.

For full examples, see ``examples/waypoint_follow.py``,
``examples/train_ppo_example.py``, and ``examples/eval_ppo_example.py``.
