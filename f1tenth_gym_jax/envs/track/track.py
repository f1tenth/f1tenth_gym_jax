import pathlib
from typing import Optional
import numpy as np
from functools import partial
import jax.numpy as jnp
import jax

from .cubic_spline import CubicSplineND


class Track:
    ss: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    vxs: np.ndarray
    centerline: CubicSplineND
    raceline: CubicSplineND
    filepath: Optional[str]
    ss: Optional[np.ndarray] = None
    psis: Optional[np.ndarray] = None
    kappas: Optional[np.ndarray] = None
    accxs: Optional[np.ndarray] = None

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        velxs: np.ndarray,
        filepath: Optional[str] = None,
        centerline: Optional[CubicSplineND] = None,
        raceline: Optional[CubicSplineND] = None,
        ss: Optional[np.ndarray] = None,
        psis: Optional[np.ndarray] = None,
        kappas: Optional[np.ndarray] = None,
        accxs: Optional[np.ndarray] = None,
        waypoints: Optional[np.ndarray] = None,
        s_frame_max: Optional[float] = None
    ):
        """
        Initialize track object.

        Parameters
        ----------
        spec : TrackSpec
            track specification
        filepath : str
            path to the track image
        occupancy_map : np.ndarray
            occupancy grid map
        centerline : Raceline, optional
            centerline of the track, by default None
        raceline : Raceline, optional
            raceline of the track, by default None
        """
        self.filepath = filepath
        self.waypoints = waypoints
        self.s_frame_max = s_frame_max

        assert xs.shape == ys.shape == velxs.shape, "inconsistent shapes for x, y, vel"

        self.n = xs.shape[0]
        self.ss = ss
        self.xs = xs
        self.ys = ys
        self.yaws = psis
        self.ks = kappas
        self.vxs = velxs
        self.axs = accxs

        # approximate track length by linear-interpolation of x,y waypoints
        # note: we could use 'ss' but sometimes it is normalized to [0,1], so we recompute it here
        self.length = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))

        self.centerline = centerline or CubicSplineND(xs, ys, psis, kappas, velxs, accxs)
        self.raceline = raceline or CubicSplineND(xs, ys, psis, kappas, velxs, accxs)
        self.s_guess = 0.0

    @staticmethod
    def from_numpy(waypoints: np.ndarray, s_frame_max, downsample_step = 1) -> Track:
        """
        Create an empty track reference line.

        Parameters
        ----------
        filepath : pathlib.Path
            path to the raceline file
        delimiter : str, optional
            delimiter used in the file, by default ";"
        downsample_step : int, optional
            downsample step for waypoints, by default 1 (no downsampling)

        Returns
        -------
        track: Track
            track object
        """
        assert (
            waypoints.shape[1] >= 7
        ), "expected waypoints as [s, x, y, psi, k, vx, ax]"
        
        ss=waypoints[::downsample_step, 0]
        xs=waypoints[::downsample_step, 1]
        ys=waypoints[::downsample_step, 2]
        yaws=waypoints[::downsample_step, 3]
        ks=waypoints[::downsample_step, 4]
        vxs=waypoints[::downsample_step, 5]
        axs=waypoints[::downsample_step, 6]

        refline = CubicSplineND(xs, ys, yaws, ks, vxs, axs)

        return Track(
            xs=xs,
            ys=ys,
            velxs=vxs,
            ss=refline.s,
            psis=yaws,
            kappas=ks,
            accxs=axs,
            filepath=None,
            raceline=refline,
            centerline=refline,
            waypoints=waypoints,
            s_frame_max=s_frame_max
        )
    
    @staticmethod
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";", downsample_step = 1) -> Track:
        """
        Create an empty track reference line.

        Parameters
        ----------
        filepath : pathlib.Path
            path to the raceline file
        delimiter : str, optional
            delimiter used in the file, by default ";"
        downsample_step : int, optional
            downsample step for waypoints, by default 1 (no downsampling)

        Returns
        -------
        track: Track
            track object
        """
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter).astype(np.float32)
        assert (
            waypoints.shape[1] >= 7
        ), "expected waypoints as [s, x, y, psi, k, vx, ax]"
        
        ss=waypoints[::downsample_step, 0]
        xs=waypoints[::downsample_step, 1]
        ys=waypoints[::downsample_step, 2]
        yaws=waypoints[::downsample_step, 3]
        ks=waypoints[::downsample_step, 4]
        vxs=waypoints[::downsample_step, 5]
        axs=waypoints[::downsample_step, 6]

        refline = CubicSplineND(xs, ys, yaws, ks, vxs, axs)

        return Track(
            xs=xs,
            ys=ys,
            velxs=vxs,
            ss=refline.s,
            psis=yaws,
            kappas=ks,
            accxs=axs,
            filepath=filepath,
            raceline=refline,
            centerline=refline,
            waypoints=waypoints,
        )
    
    @partial(jax.jit, static_argnums=(0))
    def frenet_to_cartesian(self, s, ey, ephi):
        """
        Convert Frenet coordinates to Cartesian coordinates.

        s: distance along the raceline
        ey: lateral deviation
        ephi: heading deviation

        returns:
            x: x-coordinate
            y: y-coordinate
            psi: yaw angle
        """
        s = s % self.s_frame_max
        x, y = self.centerline.calc_position(s)
        psi = self.centerline.calc_yaw(s)

        # Adjust x,y by shifting along the normal vector
        x -= ey * np.sin(psi)
        y += ey * np.cos(psi)

        # Adjust psi by adding the heading deviation
        psi += ephi
        return x, y, np.arctan2(np.sin(psi), np.cos(psi))
    
    @partial(jax.jit, static_argnums=(0))
    def vmap_frenet_to_cartesian(self, poses):
        s, ey, ephi = poses[:, 0], poses[:, 1], poses[:, 2]
        return jnp.asarray(jax.vmap(self.frenet_to_cartesian, in_axes=(0, 0, 0))(
            s, ey, ephi
            )).T
    
    @partial(jax.jit, static_argnums=(0))
    def frenet_to_cartesian(self, s, ey, ephi):
        """
        Convert Frenet coordinates to Cartesian coordinates.

        s: distance along the raceline
        ey: lateral deviation
        ephi: heading deviation

        returns:
            x: x-coordinate
            y: y-coordinate
            psi: yaw angle
        """
        s = s % self.s_frame_max
        x, y = self.centerline.calc_position(s)
        psi = self.centerline.calc_yaw(s)

        # Adjust x,y by shifting along the normal vector
        x -= ey * jnp.sin(psi)
        y += ey * jnp.cos(psi)

        # Adjust psi by adding the heading deviation
        psi += ephi

        return x, y, jnp.arctan2(jnp.sin(psi), jnp.cos(psi))
    
    @partial(jax.jit, static_argnums=(0))
    def cartesian_to_frenet(self, x, y, phi, s_guess=None):
        s, ey = self.centerline.calc_arclength(x, y, s_guess)
        # Wrap around
        s = s % self.s_frame_max

        # Use the normal to calculate the signed lateral deviation
        # normal = self.centerline._calc_normal(s)
        yaw = self.centerline.calc_yaw(s)
        normal = jnp.asarray([-jnp.sin(yaw), jnp.cos(yaw)])
        x_eval, y_eval = self.centerline.calc_position(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = jnp.sign(jnp.dot(jnp.asarray([dx, dy]), normal))
        ey = ey * distance_sign

        phi = phi - yaw

        return s, ey, jnp.arctan2(jnp.sin(phi), jnp.cos(phi))
    
    @partial(jax.jit, static_argnums=(0))
    def vmap_cartesian_to_frenet(self, poses):
        x, y, phi = poses[:, 0], poses[:, 1], poses[:, 2]
        return jnp.asarray(jax.vmap(self.cartesian_to_frenet, in_axes=(0, 0, 0))(
            x, y, phi
            )).T
    
    @partial(jax.jit, static_argnums=(0))
    def curvature(self, s):
        s = s % self.s_frame_max
        return self.centerline.calc_curvature(s)
        # return self.centerline.find_curvature_jax(s)
    
    @staticmethod
    def load_map(MAP_DIR, map_info, map_ind, config, scale=1, reverse=False, downsample_step=1):
        """
        loads waypoints
        """
        map_info = map_info[map_ind][1:]
        config.wpt_path = str(map_info[0])
        config.wpt_delim = str(map_info[1])
        config.wpt_rowskip = int(map_info[2])
        config.wpt_xind = int(map_info[3])
        config.wpt_yind = int(map_info[4])
        config.wpt_thind = int(map_info[5])
        config.wpt_vind = int(map_info[6])
        # config.s_frame_max = float(map_info[7])
        config.s_frame_max = -1
        
        
        
        waypoints = np.loadtxt(MAP_DIR + config.wpt_path, delimiter=config.wpt_delim, skiprows=config.wpt_rowskip)
        if reverse: # NOTE: reverse map
            waypoints = waypoints[::-1]
            # if map_ind == 41: waypoints[:, config.wpt_thind] = waypoints[:, config.wpt_thind] + 3.14
        # if map_ind == 41: waypoints[:, config.wpt_thind] = waypoints[:, config.wpt_thind] + np.pi / 2
        waypoints[:, config.wpt_yind] = waypoints[:, config.wpt_yind] * scale
        waypoints[:, config.wpt_xind] = waypoints[:, config.wpt_xind] * scale # NOTE: map scales
        if config.s_frame_max == -1:
            config.s_frame_max = waypoints[-1, 0]
        
        # NOTE: initialized states for forward
        if config.wpt_thind == -1:
            print('Convert to raceline format.')
            # init_theta = np.arctan2(waypoints[1, config.wpt_yind] - waypoints[0, config.wpt_yind], 
            #                         waypoints[1, config.wpt_xind] - waypoints[0, config.wpt_xind])
            waypoints = Track.centerline_to_frenet(waypoints, velocity=5.0)
            # np.save('waypoints.npy', waypoints)
            config.wpt_xind = 1
            config.wpt_yind = 2
            config.wpt_thind = 3
            config.wpt_vind = 5
        # else:
        init_theta = waypoints[0, config.wpt_thind]
        track = Track.from_numpy(waypoints, config.s_frame_max, downsample_step)
        track.waypoints_distances = np.linalg.norm(track.waypoints[1:, (1, 2)] - track.waypoints[:-1, (1, 2)], axis=1)
        
        return track, config