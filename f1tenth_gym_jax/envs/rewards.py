import jax
import jax.numpy as jnp
import chex
from functools import partial

from .f110_env import Param, State, F110Env


@partial(jax.jit, static_argnums=[1])
def keep_alive_reward_fn(current_state: State, env: F110Env):
    # reward for agents that are still alive
    rewards = 0.1 * jax.numpy.ones_like(current_state.rewards)
    rewards = rewards * current_state.collisions
    return rewards


@partial(jax.jit, static_argnums=[1])
def progress_reward_fn(current_state: State, env: F110Env):
    # reward for agents' progress in frenet frame
    agent_s = jnp.clip(current_state.frenet_states[:, 0] / env.track.length, max=1.0)
    return agent_s


@partial(jax.jit, static_argnums=[1])
def finished_reward_fn(current_state: State, env: F110Env):
    # reward for agents that have finished the track
    rewards = 10.0 * current_state.num_laps >= env.params.max_num_laps
    return rewards
