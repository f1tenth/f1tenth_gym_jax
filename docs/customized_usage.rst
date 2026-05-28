.. _custom_usage:
.. _configuration:

Environment Configuration
=========================

Configuration is split between the environment ID and optional keyword
arguments passed to ``make``. The ID chooses the map, number of agents, sensor
mode, collision mode, rewards, controls, control-step ratio, and episode length.
Keyword arguments override fields on ``f1tenth_gym_jax.envs.utils.Param``.

Environment IDs
---------------

The canonical ID format is:

.. code:: text

    {map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{longitudinal+steering}_{timestep_ratio}_{max_steps}_v0

For the default 90-second episode length, omit ``max_steps``:

.. code:: text

    {map}_{num_agents}_{scan|noscan}_{collision|nocollision}_{rewards}_{longitudinal+steering}_{timestep_ratio}_v0

Older model filenames that omit ``timestep_ratio`` are still accepted for
compatibility and use ``timestep_ratio=1``.

Examples:

.. code:: python

    from f1tenth_gym_jax import make

    env = make(
        "Spielberg_2_scan_collision_progress+alive_velocity+steeringangle_10_500_v0"
    )

    default_length_env = make(
        "Spielberg_1_scan_collision_progress_acceleration+steeringvelocity_10_v0"
    )

ID Fields
---------

``map``
    A registered map name such as ``Spielberg``, ``Monza``, or ``Spa``. Use
    ``f1tenth_gym_jax.registration.list_available_maps`` to inspect the current
    built-in and locally cached maps.

``num_agents``
    Positive integer number of cars in the environment.

``scan`` or ``noscan``
    Enables or disables laser scan observations.

``collision`` or ``nocollision``
    Enables or disables vehicle and map collision termination.

``rewards``
    ``+``-joined subset of ``time``, ``progress``, and ``alive``.

``longitudinal+steering``
    Longitudinal control is ``acceleration`` or ``velocity``. Steering control
    is ``steeringvelocity`` or ``steeringangle``.

``timestep_ratio``
    Number of dynamics integration steps per environment control step.

``max_steps``
    Maximum number of environment control steps before termination. The default
    shorthand computes ``int(90 / (timestep * timestep_ratio))``.

Actions and Bounds
------------------

The environment ID names controls as ``longitudinal+steering`` for readability,
but action vectors are ordered as ``[steering_command, longitudinal_command]``.
For example, an ``acceleration+steeringvelocity`` environment expects
``[steering_velocity, acceleration]``.

Inspect the exact bounds from the environment:

.. code:: python

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_v0"
    )
    action_space = env.action_space("agent_0")
    print(action_space.low, action_space.high)

Keyword Overrides
-----------------

Additional keyword arguments passed to ``make`` override ``Param`` fields.

.. code:: python

    env = make(
        "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_v0",
        model="ks",
        integrator="euler",
        observe_others=False,
        mu=1.0,
        timestep=0.02,
    )

Common overrides:

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Field
     - Values
     - Effect
   * - ``model``
     - ``"st"``, ``"st_smooth"``, ``"ks"``
     - Selects the single-track or kinematic bicycle dynamics.
   * - ``integrator``
     - ``"rk4"``, ``"euler"``
     - Selects the micro-step integrator.
   * - ``timestep``
     - positive float
     - Dynamics integration step in seconds.
   * - ``timestep_ratio``
     - positive integer
     - Integration micro-steps per environment step.
   * - ``observe_others``
     - boolean
     - Adds relative state of other agents to each observation.
   * - ``produce_scans``
     - boolean
     - Adds range scan beams to each observation.
   * - ``collision_on``
     - boolean
     - Enables vehicle-vehicle and vehicle-map collision checks.
   * - ``max_num_laps``
     - positive integer
     - Ends an episode after this many completed laps.
   * - ``max_steps``
     - positive integer
     - Ends an episode after this many control steps.

Vehicle parameters such as ``mu``, ``C_Sf``, ``C_Sr``, ``lf``, ``lr``,
``h``, ``m``, ``I``, steering limits, acceleration limits, vehicle dimensions,
and scan parameters are also configurable through the same keyword override
path. See :class:`f1tenth_gym_jax.envs.utils.Param` for the full set of fields.

Observation Layout
------------------

Each agent observation starts with Frenet state ``[s, ey, epsi]`` followed by
the configured Cartesian dynamics state. If ``observe_others=True``, relative
states for the other agents are appended as ``[relative_x, relative_y,
longitudinal_v, relative_psi]`` per other agent. If scans are enabled, scan
beams are appended after the state fields.

The index groups are exposed as ``env.observation_space_ind``.

Maps and Cache
--------------

Bundled maps are loaded from the installed package. Downloaded maps are stored
under ``$XDG_CACHE_HOME/f1tenth_gym_jax/maps`` by default. Set
``F1TENTH_GYM_JAX_MAP_DIR`` to use a different writable map cache.
