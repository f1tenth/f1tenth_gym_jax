.. _dynamics:

Dynamics
========

The environment integrates one dynamics model per agent at every control step.
All agents in an environment are stepped simultaneously, then scans, collisions,
rewards, and termination flags are computed from the new state.

Notation
--------

The environment action vector is always ordered as
``[steering_command, longitudinal_command]``. The configured action modes
convert commands into a steering velocity ``u_delta`` and longitudinal
acceleration ``a`` before integration:

.. math::

   u_\delta =
   \begin{cases}
   u_0, & \text{steeringvelocity mode} \\
   (u_0 - \delta) / \Delta t, & \text{steeringangle mode}
   \end{cases}

.. math::

   a =
   \begin{cases}
   u_1, & \text{acceleration mode} \\
   (u_1 - v) / \Delta t, & \text{velocity mode}
   \end{cases}

where ``u_0`` and ``u_1`` are the two action components and ``\Delta t`` is
``Param.timestep``.

Let:

.. math::

   L = l_f + l_r

where ``l_f`` and ``l_r`` are the distances from the vehicle center of gravity
to the front and rear axle.

Control Constraints
-------------------

The desired steering velocity is clipped to the steering-rate bounds and forced
to zero if the steering angle is already at a bound and the command would push
farther outside the range:

.. math::

   u_\delta =
   \begin{cases}
   0, & \delta \le s_{min} \land u_\delta^d \le 0 \\
   0, & \delta \ge s_{max} \land u_\delta^d \ge 0 \\
   sv_{min}, & u_\delta^d \le sv_{min} \\
   sv_{max}, & u_\delta^d \ge sv_{max} \\
   u_\delta^d, & \text{otherwise}
   \end{cases}

The upper acceleration limit decreases above the switching velocity:

.. math::

   a_{upper}(v) =
   \begin{cases}
   a_{max} v_{switch} / v, & v > v_{switch} \\
   a_{max}, & \text{otherwise}
   \end{cases}

The final longitudinal acceleration is:

.. math::

   a =
   \begin{cases}
   0, & v \le v_{min} \land a^d \le 0 \\
   0, & v \ge v_{max} \land a^d \ge 0 \\
   -a_{max}, & a^d \le -a_{max} \\
   a_{upper}(v), & a^d \ge a_{upper}(v) \\
   a^d, & \text{otherwise}
   \end{cases}

Kinematic Single-Track Model
----------------------------

``model="ks"`` uses the state

.. math::

   z = [x, y, \delta, v, \psi]^T

and integrates:

.. math::

   \dot{x} = v \cos(\psi)

.. math::

   \dot{y} = v \sin(\psi)

.. math::

   \dot{\delta} = u_\delta

.. math::

   \dot{v} = a

.. math::

   \dot{\psi} = \frac{v}{L}\tan(\delta)

The returned derivative vector also includes two dummy zero entries for the
control dimensions appended during integration.

Dynamic Single-Track Model
--------------------------

``model="st"`` and ``model="st_smooth"`` use the state

.. math::

   z = [x, y, \delta, v, \psi, \dot{\psi}, \beta]^T

where ``\beta`` is the vehicle slip angle. For the dynamic branch:

.. math::

   \dot{x} = v \cos(\psi + \beta)

.. math::

   \dot{y} = v \sin(\psi + \beta)

.. math::

   \dot{\delta} = u_\delta

.. math::

   \dot{v} = a

.. math::

   \dot{\psi} = \omega

with ``\omega = \dot{\psi}``. Define:

.. math::

   A_f = g l_r - a h

.. math::

   A_r = g l_f + a h

Then the yaw acceleration is:

.. math::

   \ddot{\psi} =
   \frac{\mu m}{I L}
   \left[
   l_f C_{Sf} A_f \delta
   + (l_r C_{Sr} A_r - l_f C_{Sf} A_f)\beta
   - (l_f^2 C_{Sf} A_f + l_r^2 C_{Sr} A_r)\frac{\omega}{v}
   \right]

and the slip-angle derivative is:

.. math::

   \dot{\beta} =
   \frac{\mu}{v L}
   \left[
   C_{Sf} A_f \delta
   - (C_{Sr} A_r + C_{Sf} A_f)\beta
   + (C_{Sr} A_r l_r - C_{Sf} A_f l_f)\frac{\omega}{v}
   \right]
   - \omega

Low-Speed Branch
----------------

The switching single-track model uses a kinematic branch at low speed to avoid
singular dynamic terms around ``v = 0``. It defines:

.. math::

   \hat{\beta} = \arctan\left(\tan(\delta)\frac{l_r}{L}\right)

.. math::

   \dot{\hat{\beta}} =
   \frac{1}{1 + (\tan(\delta)l_r/L)^2}
   \frac{l_r}{L \cos^2(\delta)}
   u_\delta

The low-speed branch is:

.. math::

   \dot{x} = v \cos(\psi + \hat{\beta})

.. math::

   \dot{y} = v \sin(\psi + \hat{\beta})

.. math::

   \dot{\delta} = u_\delta

.. math::

   \dot{v} = a

.. math::

   \dot{\psi} = \frac{v \cos(\hat{\beta})\tan(\delta)}{L}

.. math::

   \ddot{\psi}_{ks} =
   \frac{1}{L}
   \left[
   a\cos(\beta)\tan(\delta)
   - v\sin(\beta)\tan(\delta)\dot{\hat{\beta}}
   + \frac{v\cos(\beta)u_\delta}{\cos^2(\delta)}
   \right]

.. math::

   \dot{\beta} = \dot{\hat{\beta}}

``model="st"`` selects the low-speed branch when ``|v| < 1.5`` and the dynamic
branch otherwise. ``model="st_smooth"`` blends the branches with:

.. math::

   w(v) = \operatorname{round}(\sigma(100(0.55 - |v|)), 3)

.. math::

   f(z, u) = w(v) f_{ks}(z, u) + (1 - w(v)) f_{st}(z, u)

Integration
-----------

The environment integrates with either explicit Euler or fourth-order
Runge-Kutta. For each environment control step, the selected integrator repeats
``Param.timestep_ratio`` micro-steps of size ``Param.timestep``.

Euler:

.. math::

   z_{k+1} = z_k + \Delta t f(z_k, u_k)

RK4:

.. math::

   k_1 = f(z_k, u_k)

.. math::

   k_2 = f(z_k + \Delta t k_1 / 2, u_k)

.. math::

   k_3 = f(z_k + \Delta t k_2 / 2, u_k)

.. math::

   k_4 = f(z_k + \Delta t k_3, u_k)

.. math::

   z_{k+1} = z_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)

After every micro-step, yaw is wrapped with ``atan2(sin(psi), cos(psi))``.

Reward Terms
------------

The configured reward string selects any combination of:

``time``
    ``-Param.timestep * Param.timestep_ratio`` at each step.

``progress``
    Signed forward progress in Frenet arclength since the previous step, with
    wrap-around handling at the track boundary.

``alive``
    ``-1`` after collision and ``0`` otherwise.
