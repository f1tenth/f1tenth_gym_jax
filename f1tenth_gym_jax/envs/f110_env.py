"""
Jax jittable f1tenth_gym environment

Based on JaxMARL api which follows PettingZoo/Gymnax

Author: Hongrui Zheng
"""

# typing
from typing import Any, Dict, Optional, Tuple, Union

from .multi_agent_env import MultiAgentEnv
from .spaces import Box
from .spaces import Tuple as JaxMARLTuple
from .spaces import Dict as JaxMARLDict

# jax
import jax
import jax.numpy as jnp
import chex
from flax import struct

# numpy
import numpy as np

# local
from .action import CarAction, from_single_to_multi_action_space
from .integrator import IntegratorType
from .rendering import make_renderer

from .track import Track

# base classes
from .base_classes import Simulator, DynamicModel
from .observation import observation_factory, Observation
from .reset import make_reset_fn
from .track import Track
from .utils import deep_update

# other
from functools import partial


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
    cartesian_states: chex.Array  # [n_agent, [x, y, delta, v, psi, psi_dot, beta]]
    frenet_states: chex.Array  # [n_agent, [s, ey, epsi]]
    collisions: chex.Array  # [n_agent, n_agent + 1]

    # laser scans TODO: might not need to be part of the state since doesn't depend on previous
    scans: chex.Array  # [n_agent, n_rays]

    # race stuff
    num_laps: chex.Array # [n_agent, ]
    


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
    integrator: str = "rk4"  # dynamics integrator
    model: str = "st"  # dynamics model type
    produce_scans: bool = False # whether to turn on laser scan


class F110Env(MultiAgentEnv):
    """
    JAX compatible gym environment for F1TENTH

    Args:
        kwargs:
            num_agents (int, default=2): number of agents in the environment
            map (str, default='vegas'): name of the map used for the environment.
            params (Parm): vehicle parameters.

    """

    def __init__(self, num_agents: int = 1, params: Param = Param(), **kwargs):
        super().__init__()

        # Configuration
        # self.config = self.default_config()
        # self.configure(config)

        # self.seed = self.config["seed"]
        # self.map = self.config["map"]
        # self.params = self.config["params"]
        # self.num_agents = self.config["num_agents"]
        # self.timestep = self.config["timestep"]
        # self.ego_idx = self.config["ego_idx"]
        # self.integrator = IntegratorType.from_string(self.config["integrator"])
        # self.model = DynamicModel.from_string(self.config["model"])
        # self.observation_config = self.config["observation_config"]
        # self.action_type = CarAction(self.config["control_input"], params=self.params)

        # radius to consider done
        # self.start_thresh = 0.5  # 10cm

        # env states
        # self.poses_x = []
        # self.poses_y = []
        # self.poses_theta = []
        # self.collisions = np.zeros((self.num_agents,))

        self.num_agents = num_agents

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

        # loop completion
        # self.near_start = True
        # self.num_toggles = 0

        # race info
        # self.lap_times = np.zeros((self.num_agents,))
        # self.lap_counts = np.zeros((self.num_agents,))
        # self.current_time = 0.0

        # finish line info
        # self.num_toggles = 0
        # self.near_start = True
        # self.near_starts = np.array([True] * self.num_agents)
        # self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(
            self.params,
            self.num_agents,
            self.seed,
            time_step=self.timestep,
            integrator=self.integrator,
            model=self.model,
            action_type=self.action_type,
        )
        self.sim.set_map(self.map)

        if isinstance(self.map, Track):
            self.track = self.map
        else:
            self.track = Track.from_track_name(
                self.map
            )  # load track in gym env for convenience

        # observations
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        assert (
            "type" in self.observation_config
        ), "observation_config must contain 'type' key"
        self.observation_type = observation_factory(env=self, **self.observation_config)
        self.observation_space = self.observation_type.space()

        # action space
        self.action_space = from_single_to_multi_action_space(
            self.action_type.space, self.num_agents
        )

        # reset modes
        self.reset_fn = make_reset_fn(
            **self.config["reset_config"], track=self.track, num_agents=self.num_agents
        )

        # stateful observations for rendering
        # add choice of colors (same, random, ...)
        self.render_obs = None
        self.render_mode = render_mode

        # match render_fps to integration timestep
        self.metadata["render_fps"] = int(1.0 / self.timestep)
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10  # boost fps by 10x
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        # TODO: step f1tenth env
        pass

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return bool(done), self.toggle_list >= 4

    def _update_state(self):
        """
        Update the env's states according to observations.
        """
        self.poses_x = self.sim.agent_poses[:, 0]
        self.poses_y = self.sim.agent_poses[:, 1]
        self.poses_theta = self.sim.agent_poses[:, 2]
        self.collisions = self.sim.collisions

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        self.sim.step(action)

        # observation
        obs = self.observation_type.observe()

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state()

        # rendering observation
        self.render_obs = {
            "ego_idx": self.sim.ego_idx,
            "poses_x": self.sim.agent_poses[:, 0],
            "poses_y": self.sim.agent_poses[:, 1],
            "poses_theta": self.sim.agent_poses[:, 2],
            "steering_angles": self.sim.agent_steerings,
            "lap_times": self.lap_times,
            "lap_counts": self.lap_counts,
            "collisions": self.sim.collisions,
            "sim_time": self.current_time,
        }

        # check done
        done, toggle_list = self._check_done()
        truncated = False
        info = {"checkpoint_done": toggle_list}

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            seed: random seed for the reset
            options: dictionary of options for the reset containing initial poses of the agents

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        if options is not None and "poses" in options:
            poses = options["poses"]
        else:
            poses = self.reset_fn.sample()

        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be a numpy array of shape (num_agents, 3)"

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        return obs, info

    def update_map(self, map_name: str):
        """
        Updates the map used by simulation

        Args:
            map_name (str): name of the map

        Returns:
            None
        """
        self.sim.set_map(map_name)
        self.track = Track.from_track_name(map_name)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        self.renderer.add_renderer_callback(callback_func)

    def render(self, mode="human"):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        # NOTE: separate render (manage render-mode) from render_frame (actual rendering with pyglet)

        if self.render_mode not in self.metadata["render_modes"]:
            return

        self.renderer.update(state=self.render_obs)
        return self.renderer.render()

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        if self.renderer is not None:
            self.renderer.close()
        super().close()
