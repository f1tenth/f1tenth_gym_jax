# other
from typing import Callable
from functools import partial

# jax
import jax
import chex

# local
from .f110_env import Param


@partial(jax.jit, static_argnums=[0, 2])
def integrate_rk4(f: Callable, x_and_u: chex.Array, params: Param) -> chex.Array:
    k1 = f(x_and_u, params)
    k2_state = x_and_u + params.timestep * (k1 / 2)
    k2 = f(k2_state, params)
    k3_state = x_and_u + params.timestep * (k2 / 2)
    k3 = f(k3_state, params)
    k4_state = x_and_u + params.timestep * k3
    k4 = f(k4_state, params)
    # dynamics integration
    x_and_u = x_and_u + params.timestep * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_and_u


@partial(jax.jit, static_argnums=[3, 4])
def integrate_euler(f: Callable, x_and_u: chex.Array, params: Param) -> chex.Array:
    dstate = f(x_and_u, params)
    x_and_u = x_and_u + params.timestep * dstate
    return x_and_u
