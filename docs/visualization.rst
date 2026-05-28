.. _visualization:

Visualization
=============

Rollout visualization is web-based. ``WebRenderer`` writes a standalone HTML
file that can be opened in any browser; it does not require a Python process,
desktop display server, Qt, or OpenGL after the file is generated.

The dashboard includes:

* summary cards for batched rollout count, step count, agents, duration, speed,
  and distance
* labeled overview traces for every rollout and agent
* a trajectory playback canvas with vehicle labels
* a timestep scrubber
* a speed multiplier scrubber that defaults to ``1.0x`` actual environment time
* a per-rollout statistics table

Generate a Dashboard
--------------------

The quickest path is the standalone example:

.. code:: bash

    uv run python examples/render_dashboard.py --steps 120 --output /tmp/rollout.html

For a batched waypoint-following rollout:

.. code:: bash

    uv run python examples/waypoint_follow.py \
      --num-agents 3 \
      --num-envs 10 \
      --steps 500 \
      --render-output /tmp/f1tenth_dashboard.html

Use ``--no-render`` on examples that support it when only benchmark or training
output is needed.

Render From Python
------------------

``WebRenderer.render`` accepts trajectory arrays in one of these layouts:

``(steps, envs, agents, states)``
    Default layout produced by ``jax.lax.scan`` around ``jax.vmap``.

``(envs, steps, agents, states)``
    Pass ``trajectory_layout="batch_major"`` to ``WebRenderer``.

``(steps, agents, states)``
    Single rollout with multiple agents.

``(steps, states)``
    Single rollout with one agent.

The state vector must include at least ``[x, y, steering_angle, velocity, yaw]``.

.. code:: python

    import pathlib

    import jax
    import jax.numpy as jnp
    import numpy as np

    from f1tenth_gym_jax import make
    from f1tenth_gym_jax.envs.rendering import WebRenderer

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_200_v0"
    )
    key = jax.random.key(0)
    _, state = env.reset(key)

    def step(carry, unused):
        state, key = carry
        key, step_key = jax.random.split(key)
        actions = {"agent_0": jnp.array([0.0, 1.0])}
        _, state, rewards, dones, infos = env.step(step_key, state, actions)
        return (state, key), state.cartesian_states

    (state, key), trajectory = jax.lax.scan(step, (state, key), None, length=200)

    WebRenderer(env).render(
        np.asarray(trajectory),
        output_path=pathlib.Path("/tmp/f1tenth_dashboard.html"),
        metadata={"controller": "constant acceleration"},
    )

Real-Time Playback
------------------

The dashboard playback interval is based on
``env.params.timestep * env.params.timestep_ratio`` unless ``render_fps`` is
provided to ``WebRenderer``. At the default speed multiplier of ``1.0x``, one
dashboard second corresponds to one simulated second.

To visualize a trajectory recorded at a different cadence, pass an explicit
``render_fps``:

.. code:: python

    renderer = WebRenderer(env, render_fps=20.0)
    renderer.render(trajectory, output_path="/tmp/twenty_hz_rollout.html")

Opening the Browser
-------------------

By default the renderer only writes the HTML file. Set ``open_browser=True`` to
ask Python to open the generated file with the system browser:

.. code:: python

    WebRenderer(env, open_browser=True).render(trajectory)
