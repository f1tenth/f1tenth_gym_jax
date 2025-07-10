from functools import partial
from typing import Dict, List, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from .multi_agent_env import MultiAgentEnv


@struct.dataclass
class State:
    """
    Basic Jittable state for cars
    """

    # gym stuff
    rewards: chex.Array  # [n_agent, ]
    done: chex.Array  # [n_agent, ]
    step: int

    # dynamic states
    cartesian_states: (
        chex.Array
    )  # [n_agent, [x, y, delta, v, psi, (psi_dot, beta)]], extra states for st in ()
    frenet_states: chex.Array  # [n_agent, [s, ey, epsi]]
    collisions: chex.Array  # [n_agent,]

    # race stuff
    num_laps: chex.Array  # [n_agent, ]

    # laser scans TODO: might not need to be part of the state since doesn't depend on previous
    scans: chex.Array  # [n_agent, n_rays]

    # winding vector
    prev_winding_vector: chex.Array  # [n_agent, 2]
    last_accumulated_angles: chex.Array  # [n_agent, 1]
    accumulated_angles: chex.Array  # [n_agent, 1]


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


@struct.dataclass
class Param:
    """
    Default jittable params for dynamics
    """

    mu: float = 1.0489  # surface friction coefficient
    C_Sf: float = 4.718  # Cornering stiffness coefficient, front
    C_Sr: float = 5.4562  # Cornering stiffness coefficient, rear
    lf: float = 0.15875  # Distance from center of gravity to front axle
    lr: float = 0.17145  # Distance from center of gravity to rear axle
    h: float = 0.074  # Height of center of gravity
    m: float = 3.74  # Total mass of the vehicle
    I: float = 0.04712  # Moment of inertial of the entire vehicle about the z axis
    s_min: float = -0.4189  # Minimum steering angle constraint
    s_max: float = 0.4189  # Maximum steering angle constraint
    sv_min: float = -3.2  # Minimum steering velocity constraint
    sv_max: float = 3.2  # Maximum steering velocity constraint
    v_switch: float = (
        7.319  # Switching velocity (velocity at which the acceleration is no longer able to #spin)
    )
    a_max: float = 9.51  # Maximum longitudinal acceleration
    v_min: float = -5.0  # Minimum longitudinal velocity
    v_max: float = 20.0  # Maximum longitudinal velocity
    width: float = 0.31  # width of the vehicle in meters
    length: float = 0.58  # length of the vehicle in meters
    timestep: float = 0.01  # physical time steps of the dynamics model
    longitudinal_action_type: str = "acceleration"  # speed or acceleration
    steering_action_type: str = (
        "steering_velocity"  # steering_angle or steering_velocity
    )
    integrator: str = "rk4"  # dynamics integrator
    model: str = "st"  # dynamics model type
    produce_scans: bool = False  # whether to turn on laser scan
    collision_on: bool = True  # whether to turn on collision detection
    theta_dis: int = 2000  # number of discretization in theta, scan param
    fov: float = 4.7  # field of view of the scan, scan param
    num_beams: int = 64  # number of beams in each scan, scan param
    eps: float = 0.01  # epsilon to stop ray marching, scan param
    max_range: float = 10.0  # max range of scan, scan param
    observe_others: bool = True  # whether can observe other agents
    map_name: str = "Spielberg"  # map for environment
    max_num_laps: int = 1  # maximum number of laps to run before done
    max_steps: int = int(120 / timestep)  # maximum number of steps to run before done
    reward_type: str = "progress" # reward types


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


class Wrapper:
    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


class LogWrapper(Wrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


class WorldStateWrapper(Wrapper):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: State, actions: dict):
        obs, env_state, reward, done, info = self._env.step(key, state, actions)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def world_state(self, obs: Dict[str, chex.Array]):
        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs
