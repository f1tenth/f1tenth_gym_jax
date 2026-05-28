.. _basic_usage:

Basic Usage
===========

The environment API is JAX-native. Create environments with
``f1tenth_gym_jax.make`` and step them with explicit PRNG keys, immutable
state, and per-agent action dictionaries.

Reset and Step
--------------

The shortest single-agent loop is:

.. code:: python

    import jax
    import jax.numpy as jnp

    from f1tenth_gym_jax import make

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_100_v0"
    )

    key = jax.random.key(0)
    obs, state = env.reset(key)

    # Action order is [steering_command, longitudinal_command].
    actions = {"agent_0": jnp.array([0.0, 1.0])}
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, infos = env.step(step_key, state, actions)

``reset`` returns an observation dictionary and a JAX state object. ``step``
returns observations, the next state, per-agent rewards, per-agent done flags
plus ``"__all__"``, and an info dictionary.

Action vectors are ordered as ``[steering_command, longitudinal_command]``.
For the environment above, that means ``[steering_velocity, acceleration]``.

Multi-Agent Actions
-------------------

Multi-agent environments use one action vector per agent. The agent names are
``agent_0`` through ``agent_{N-1}``.

.. code:: python

    env = make(
        "Spielberg_2_noscan_collision_progress+alive_acceleration+steeringvelocity_1_500_v0"
    )
    key = jax.random.key(1)
    obs, state = env.reset(key)

    actions = {
        "agent_0": jnp.array([0.0, 1.0]),
        "agent_1": jnp.array([0.0, 0.8]),
    }
    obs, state, rewards, dones, infos = env.step(key, state, actions)

The per-agent observation and action spaces are available from
``env.observation_space(agent)`` and ``env.action_space(agent)``.

Batched Rollouts
----------------

The environment is designed for ``jax.vmap`` and ``jax.lax.scan``. The example
below runs eight environments in parallel for 100 control steps.

.. code:: python

    import jax
    import jax.numpy as jnp

    from f1tenth_gym_jax import make

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_100_v0"
    )
    num_envs = 8
    key = jax.random.key(0)

    reset_keys = jax.random.split(key, num_envs)
    obs, states = jax.vmap(env.reset)(reset_keys)

    def step(carry, unused):
        states, key = carry
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        actions = {"agent_0": jnp.zeros((num_envs, 2))}
        obs, states, rewards, dones, infos = jax.vmap(env.step)(
            step_keys, states, actions
        )
        return (states, key), states.cartesian_states

    (states, key), trajectory = jax.lax.scan(step, (states, key), None, length=100)

``trajectory`` has shape ``(steps, envs, agents, state_dim)`` and can be passed
directly to the web dashboard renderer.

Example Scripts
---------------

The repository includes runnable examples:

``examples/run_in_empty_track.py``
    Minimal single-agent rollout.

``examples/waypoint_follow.py``
    Batched pure-pursuit waypoint following with optional dashboard output.

``examples/mppi_example.py``
    Batched MPPI control example.

``examples/render_dashboard.py``
    Small standalone rollout that writes an HTML dashboard.

``examples/train_ppo_example.py`` and ``examples/eval_ppo_example.py``
    PPO training and evaluation entry points.
