"""
Built off Gymnax spaces.py, this module contains jittable classes for action and observation spaces.

From JaxMARL: https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/spaces.py

Added here since only using space and env abstract classes but not as a full dep
"""

from collections import OrderedDict
from typing import Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp


class Space(object):
    """
    Minimal jittable class for abstract jaxmarl space.
    """

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    def contains(self, x: jnp.int_) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """
    Minimal jittable class for discrete gymnax spaces.
    """

    def __init__(self, num_categories: int, dtype=jnp.int32):
        if num_categories < 1:
            raise ValueError("Discrete spaces must have at least one category.")
        self.n = num_categories
        self.shape = ()
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        x = jnp.asarray(x)
        if x.shape != self.shape:
            return False
        if not jnp.issubdtype(x.dtype, jnp.integer):
            return False
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class MultiDiscrete(Space):
    """
    Minimal jittable class for multi-discrete gymnax spaces.
    """

    def __init__(self, num_categories: Sequence[int], dtype=jnp.int32):
        """Num categories is the number of cat actions for each dim, [2,2,2]=2 actions x 3 dim"""
        if any(n < 1 for n in num_categories):
            raise ValueError("MultiDiscrete categories must all be positive.")
        self.num_categories = jnp.asarray(num_categories, dtype=dtype)
        self.shape = (len(num_categories),)
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng,
            shape=self.shape,
            minval=0,
            maxval=self.num_categories,
            dtype=self.dtype,
        )

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        x = jnp.asarray(x)
        if x.shape != self.shape:
            return False
        if not jnp.issubdtype(x.dtype, jnp.integer):
            return False
        range_cond = jnp.logical_and(x >= 0, x < self.num_categories)
        return jnp.all(range_cond)


class Box(Space):
    """
    Minimal jittable class for array-shaped gymnax spaces.
    """

    def __init__(
        self,
        low: float,
        high: float,
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        self.low = jnp.broadcast_to(jnp.asarray(low, dtype=dtype), shape)
        self.high = jnp.broadcast_to(jnp.asarray(high, dtype=dtype), shape)
        if bool(jnp.any(self.low > self.high)):
            raise ValueError(
                "Box low values must be less than or equal to high values."
            )

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample a random value, supporting bounded and unbounded dimensions."""
        uniform_rng, normal_rng, exp_rng = jax.random.split(rng, 3)
        bounded = jnp.logical_and(jnp.isfinite(self.low), jnp.isfinite(self.high))
        lower_bounded = jnp.logical_and(jnp.isfinite(self.low), jnp.isposinf(self.high))
        upper_bounded = jnp.logical_and(jnp.isneginf(self.low), jnp.isfinite(self.high))

        uniform_sample = jax.random.uniform(
            uniform_rng,
            shape=self.shape,
            minval=jnp.where(bounded, self.low, 0.0),
            maxval=jnp.where(bounded, self.high, 1.0),
        )
        normal_sample = jax.random.normal(normal_rng, shape=self.shape)
        exp_sample = jax.random.exponential(exp_rng, shape=self.shape)

        sample = jnp.where(
            bounded,
            uniform_sample,
            jnp.where(
                lower_bounded,
                self.low + exp_sample,
                jnp.where(upper_bounded, self.high - exp_sample, normal_sample),
            ),
        )
        return sample.astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        x = jnp.asarray(x)
        if x.shape != self.shape:
            return False
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond


class Dict(Space):
    """Minimal jittable class for dictionary of simpler jittable spaces."""

    def __init__(self, spaces: dict):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, rng: chex.PRNGKey) -> dict:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return OrderedDict(
            [
                (k, self.spaces[k].sample(key_split[i]))
                for i, k in enumerate(self.spaces)
            ]
        )

    def contains(self, x: jnp.int_) -> bool:
        """Check whether dimensions of object are within subspace."""
        if not isinstance(x, dict) or set(x) != set(self.spaces):
            return False
        out_of_space = False
        for k, space in self.spaces.items():
            out_of_space = jnp.logical_or(
                out_of_space, jnp.logical_not(space.contains(x[k]))
            )
        return jnp.logical_not(out_of_space)


class Tuple(Space):
    """Minimal jittable class for tuple (product) of jittable spaces."""

    def __init__(self, spaces: Union[tuple, list]):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, rng: chex.PRNGKey) -> Tuple[chex.Array]:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return tuple(
            [space.sample(key_split[i]) for i, space in enumerate(self.spaces)]
        )

    def contains(self, x: jnp.int_) -> bool:
        """Check whether dimensions of object are within subspace."""
        if not isinstance(x, tuple) or len(x) != self.num_spaces:
            return False
        out_of_space = False
        for value, space in zip(x, self.spaces):
            out_of_space = jnp.logical_or(
                out_of_space, jnp.logical_not(space.contains(value))
            )
        return jnp.logical_not(out_of_space)
