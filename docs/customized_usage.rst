.. _custom_usage:

Customized Usage
================

Environment IDs
---------------

``make`` parses environment IDs with this pattern:

.. code:: text

    {map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{longitudinal+steering}_{timestep_ratio}_{max_steps}_v0

For the default episode length, use the shorthand form without the
``max_steps`` field:

.. code:: text

    {map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{longitudinal+steering}_{timestep_ratio}_v0

For example:

.. code:: python

    from f1tenth_gym_jax import make

    env = make(
        "Spielberg_2_scan_collision_progress+alive_velocity+steeringangle_10_500_v0"
    )

    default_length_env = make(
        "Spielberg_1_scan_collision_progress+alive_velocity+steeringangle_10_v0"
    )

Valid longitudinal controls are ``acceleration`` and ``velocity``. Valid
steering controls are ``steeringvelocity`` and ``steeringangle``. Rewards can be
combined with ``+`` from ``time``, ``progress``, and ``alive``.

The environment ID names controls as ``longitudinal+steering`` for readability,
but action vectors are ordered as ``[steering_command, longitudinal_command]``.
For example, an ``acceleration+steeringvelocity`` environment expects
``[steering_velocity, acceleration]``. Use ``env.action_space(agent)`` to inspect
the exact bounds for each action component.

Older model filenames that omit ``timestep_ratio`` are still accepted for
compatibility and use ``timestep_ratio=1``.

Parameter Overrides
-------------------

Additional keyword arguments passed to ``make`` override fields on
``f1tenth_gym_jax.envs.utils.Param``.

.. code:: python

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_v0",
        model="ks",
        integrator="euler",
        observe_others=False,
        mu=1.0,
    )

Vectorized Rollouts
-------------------

The environment is designed for JAX transforms. Use ``jax.vmap`` over keys,
state, and action dictionaries for batched simulation.

.. code:: python

    import jax
    import jax.numpy as jnp

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_100_v0"
    )
    keys = jax.random.split(jax.random.key(0), 8)
    obs, states = jax.vmap(env.reset)(keys)

    actions = {"agent_0": jnp.zeros((8, 2))}
    step_keys = jax.random.split(jax.random.key(1), 8)
    obs, states, rewards, dones, infos = jax.vmap(env.step)(step_keys, states, actions)
