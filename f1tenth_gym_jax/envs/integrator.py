# other
from typing import Callable
from functools import partial

# jax
import jax
import chex

# local
from .f110_env import Param


@partial(jax.jit, static_argnums=[3, 4])
def integrate_rk4(
    f: Callable, x: chex.Array, u: chex.Array, dt: float, params: Param
) -> chex.Array:
    k1 = f(
        x,
        u,
        params.mu,
        params.C_Sf,
        params.C_Sr,
        params.lf,
        params.lr,
        params.h,
        params.m,
        params.I,
        params.s_min,
        params.s_max,
        params.sv_min,
        params.sv_max,
        params.v_switch,
        params.a_max,
        params.v_min,
        params.v_max,
    )

    k2_state = x + dt * (k1 / 2)

    k2 = f(
        k2_state,
        u,
        params.mu,
        params.C_Sf,
        params.C_Sr,
        params.lf,
        params.lr,
        params.h,
        params.m,
        params.I,
        params.s_min,
        params.s_max,
        params.sv_min,
        params.sv_max,
        params.v_switch,
        params.a_max,
        params.v_min,
        params.v_max,
    )

    k3_state = x + dt * (k2 / 2)

    k3 = f(
        k3_state,
        u,
        params.mu,
        params.C_Sf,
        params.C_Sr,
        params.lf,
        params.lr,
        params.h,
        params.m,
        params.I,
        params.s_min,
        params.s_max,
        params.sv_min,
        params.sv_max,
        params.v_switch,
        params.a_max,
        params.v_min,
        params.v_max,
    )

    k4_state = x + dt * k3

    k4 = f(
        k4_state,
        u,
        params.mu,
        params.C_Sf,
        params.C_Sr,
        params.lf,
        params.lr,
        params.h,
        params.m,
        params.I,
        params.s_min,
        params.s_max,
        params.sv_min,
        params.sv_max,
        params.v_switch,
        params.a_max,
        params.v_min,
        params.v_max,
    )

    # dynamics integration
    x = x + dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@partial(jax.jit, static_argnums=[3, 4])
def integrate_euler(
    f: Callable, x: chex.Array, u: chex.Array, dt: float, params: Param
) -> chex.Array:
    dstate = f(
        x,
        u,
        params.mu,
        params.C_Sf,
        params.C_Sr,
        params.lf,
        params.lr,
        params.h,
        params.m,
        params.I,
        params.s_min,
        params.s_max,
        params.sv_min,
        params.sv_max,
        params.v_switch,
        params.a_max,
        params.v_min,
        params.v_max,
    )
    x = x + dt * dstate
    return x
