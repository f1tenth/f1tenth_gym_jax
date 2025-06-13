import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.dynamic_models import (
    vehicle_dynamics_ks,
    vehicle_dynamics_st_smooth,
)


class MPPI:
    def __init__(self, config, env, rng, temperature=0.01, damping=0.001, track=None):
        self.config = config
        self.env = env
        self.rng = rng
        self.temperature = temperature
        self.damping = damping
        self.track = track

        self.n_iterations = config.n_iterations
        self.n_steps = config.n_steps
        self.n_samples = config.n_samples
        self.a_std = jnp.array(config.control_sample_std)
        self.a_cov_shift = config.a_cov_shift
        self.adaptive_covariance = (
            config.adaptive_covariance and self.n_iterations > 1
        ) or self.a_cov_shift
        self.a_shape = config.control_dim

        self.init_state()
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))

    def init_state(self):
        dim_a = jnp.prod(self.a_shape)
        self.rng, _rng = jax.random.split(self.rng)
        self.a_opt = 0.0 * jax.random.uniform(_rng, shape=(self.n_steps, dim_a))

        # a_cov: [n_steps, dim_a, dim_a]
        if self.a_cov_shift:
            self.a_cov = (self.a_std**2) * jnp.tile(
                jnp.eye(dim_a), (self.n_steps, 1, 1)
            )
            self.a_cov_init = self.a_cov.copy()
        else:
            self.a_cov = None
            self.a_cov_init = None

    @partial(jax.jit, static_argnums=(0))
    def shift_prev_opt(self, a_opt, a_cov):
        a_opt = jnp.concatenate(
            [a_opt[1:, :], jnp.expand_dims(jnp.zeros((self.a_shape,)), axis=0)]
        )  # [n_steps, a_shape]
        if self.a_cov_shift:
            a_cov = jnp.concatenate(
                [
                    a_cov[1:, :],
                    jnp.expand_dims((self.a_std**2) * jnp.eye(self.a_shape), axis=0),
                ]
            )
        else:
            a_cov = self.a_cov_init
        return a_opt, a_cov


    @partial(jax.jit, static_argnums=(0))
    def iteration_step(self, a_opt, a_cov, rng_da, env_state, reference_traj):
        rng_da, rng_da_split1, rng_da_split2 = jax.random.split(rng_da, 3)
        da = jax.random.truncated_normal(
            rng_da,
            -jnp.ones_like(a_opt) * self.a_std - a_opt,
            jnp.ones_like(a_opt) * self.a_std - a_opt,
            shape=(self.n_samples, self.n_steps, self.a_shape)
        )  # [n_samples, n_steps, dim_a]

        actions = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
        states = jax.vmap(self.rollout, in_axes=(0, None, None))(
            actions, env_state, rng_da_split1
        )
        
        if self.config.state_predictor in self.config.cartesian_models:
            reward = jax.vmap(self.env.reward_fn_xy, in_axes=(0, None))(
                states, reference_traj
            )
        else:
            reward = jax.vmap(self.env.reward_fn_sey, in_axes=(0, None))(
                states, reference_traj
            ) # [n_samples, n_steps]          
        
        R = jax.vmap(self.returns)(reward) # [n_samples, n_steps], pylint: disable=invalid-name
        w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [n_steps, dim_a]
        a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)  # [n_steps, dim_a]
        if self.adaptive_covariance:
            a_cov = jax.vmap(jax.vmap(jnp.outer))(
                da, da
            )  # [n_samples, n_steps, a_shape, a_shape]
            a_cov = jax.vmap(jnp.average, (1, None, 1))(
                a_cov, 0, w
            )  # a_cov: [n_steps, a_shape, a_shape]
            a_cov = a_cov + jnp.eye(self.a_shape)*0.00001 # prevent loss of rank when one sample is heavily weighted
            
        if self.config.render:
            traj_opt = self.rollout(a_opt, env_state, rng_da_split2)
        else:
            traj_opt = states[0]
            
        return a_opt, a_cov, states, traj_opt
    




# TODOs:
# 1. uniform sampling instead of truncated normal
# 2. remove inferenv, use explicit functions for dynamics and rewards
# 3. 