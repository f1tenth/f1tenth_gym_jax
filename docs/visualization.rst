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
* pan and zoom controls for the overview and playback canvases
* visualization toggles for map layers, labels, trajectories, vehicles, and
  visible agents
* optional artifact overlays, such as sampled controller trajectories or
  reference paths
* a playback camera selector that can center on a selected agent
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

Open or Host the Dashboard
--------------------------

If the browser is running on the same machine, open the generated HTML file
directly:

.. code:: bash

    xdg-open /tmp/f1tenth_dashboard.html

``WebRenderer`` can also ask Python to open the generated file with the system
browser:

.. code:: python

    WebRenderer(env, open_browser=True).render(trajectory)

If the browser is running on another machine, serve the output directory from
the remote host:

.. code:: bash

    uv run python -m http.server 8766 --bind 0.0.0.0 --directory /tmp

Then open the dashboard from the remote browser:

.. code:: text

    http://remote-host:8766/f1tenth_dashboard.html

Use the host name or IP address that reaches the remote host. Stop the server
with ``Ctrl+C`` when it is no longer needed.

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

Artifact Overlays
-----------------

``WebRenderer.render`` accepts optional ``artifacts`` for extra paths that
should be drawn on top of the playback view. This is useful for debugging
controllers that produce candidate plans in addition to the final executed
trajectory.

The MPPI example can write sampled plans, the selected plan, and the reference
trajectory into the dashboard:

.. code:: bash

    uv run python examples/mppi_example.py \
      --steps 300 \
      --num-samples 64 \
      --horizon 12 \
      --no-plots \
      --render-mppi-artifacts \
      --artifact-stride 5 \
      --artifact-max-samples 24 \
      --render-output /tmp/mppi_dashboard.html

The artifact controls appear under ``Visualization Options``. The sampled
trajectories are colorized by their MPPI cost, while the selected trajectory and
reference trajectory are drawn as separate labeled overlays.

For custom overlays, pass a dictionary with an ``overlays`` list:

.. code:: python

    artifacts = {
        "overlays": [
            {
                "id": "candidate-samples",
                "label": "Candidate samples",
                "type": "sample_paths",
                "points": sample_paths,
                "values": sample_costs,
                "value_label": "cost",
                "value_mode": "lower_better",
            },
            {
                "id": "selected-plan",
                "label": "Selected plan",
                "type": "paths",
                "points": selected_plan,
                "color": "#dc2626",
                "line_width": 3.0,
            },
        ]
    }

    WebRenderer(env).render(
        trajectory,
        output_path="/tmp/controller_debug.html",
        artifacts=artifacts,
    )

``paths`` overlays use shape ``(steps, rollouts, agents, points, state)``.
``sample_paths`` overlays use shape
``(steps, rollouts, agents, samples, points, state)``. The renderer extracts
``x`` and ``y`` from columns ``0`` and ``1`` by default; pass ``x_index`` and
``y_index`` on an overlay to use different columns.

Large artifact arrays can make a standalone HTML file very large. Prefer
downsampling with ``step_indices`` or example options such as
``--artifact-stride``, ``--artifact-max-steps``, and
``--artifact-max-samples``.

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
