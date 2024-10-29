import pathlib
from typing import Optional
import numpy as np
from functools import partial
import jax.numpy as jnp
import jax
import yaml
from PIL import Image
from PIL.Image import Transpose

from .cubic_spline import CubicSplineND
from .utils import find_track_dir


class Track:
    ss: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    vxs: np.ndarray
    occ_map: np.ndarray
    resolution: float
    ox: float
    oy: float
    oyaw: float
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
        occ_map: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        oyaw: float,
        velxs: np.ndarray = None,
        filepath: Optional[str] = None,
        centerline: Optional[CubicSplineND] = None,
        raceline: Optional[CubicSplineND] = None,
        ss: Optional[np.ndarray] = None,
        psis: Optional[np.ndarray] = None,
        kappas: Optional[np.ndarray] = None,
        accxs: Optional[np.ndarray] = None,
        waypoints: Optional[np.ndarray] = None,
        s_frame_max: Optional[float] = None,
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
        # self.s_frame_max = s_frame_max

        assert xs.shape == ys.shape, "inconsistent shapes for x, y"

        self.n = xs.shape[0]
        self.ss = ss
        self.xs = xs
        self.ys = ys
        self.yaws = psis
        self.ks = kappas
        self.vxs = velxs
        self.axs = accxs
        self.occ_map = occ_map
        self.resolution = resolution
        self.ox = ox
        self.oy = oy
        self.oyaw = oyaw

        # approximate track length by linear-interpolation of x,y waypoints
        # note: we could use 'ss' but sometimes it is normalized to [0,1], so we recompute it here
        self.length = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))
        self.s_frame_max = self.length

        self.centerline = centerline or CubicSplineND(
            xs, ys, psis, kappas, velxs, accxs
        )
        self.raceline = raceline or CubicSplineND(xs, ys, psis, kappas, velxs, accxs)
        self.s_guess = 0.0

    @staticmethod
    def from_track_name(map_name: str):
        # find track dir
        track_dir = find_track_dir(map_name)
        # load map yaml
        map_metadata = yaml.safe_load(
            open(str(track_dir / f"{track_dir.stem}_map.yaml"), "r")
        )
        resolution = map_metadata["resolution"]
        ox = map_metadata["origin"][0]
        oy = map_metadata["origin"][1]
        oyaw = map_metadata["origin"][2]

        # load occupancy grid
        map_filename = pathlib.Path(map_metadata["image"])
        image = Image.open(track_dir / str(map_filename)).transpose(
            Transpose.FLIP_TOP_BOTTOM
        )
        occ_map = np.array(image).astype(np.float32)
        occ_map[occ_map <= 128] = 0.0
        occ_map[occ_map > 128] = 255.0

        # if exist load centerline
        if (track_dir / f"{map_name}_centerline.csv").exists():
            # get centerline spline here
            cl_data = np.loadtxt(
                track_dir / f"{map_name}_centerline.csv", delimiter=","
            )
            assert (
                cl_data.shape[1] == 4
            ), "expected centerline columns as [x, y, w_left, w_right]"
            cl_xs, cl_ys = cl_data[:, 0], cl_data[:, 1]
            cl_xs = np.append(cl_xs, cl_xs[0])
            cl_ys = np.append(cl_ys, cl_ys[0])
            centerline = CubicSplineND(cl_xs, cl_ys)
        else:
            raise ValueError("At least centerline file is expected to construct track.")
            centerline = None

        # if exist loat raceline
        if (track_dir / f"{map_name}_raceline.csv").exists():
            # get raceline spline here
            rl_data = np.loadtxt(track_dir / f"{map_name}_raceline.csv", delimiter=";")
            assert (
                rl_data.shape[1] == 7
            ), "expected centerline columns as [s, x, y, psi, kappa, vx, ax]"
            xs, ys, psis, kappas, vxs, axs = (
                rl_data[:, 1],
                rl_data[:, 2],
                rl_data[:, 3],
                rl_data[:, 4],
                rl_data[:, 5],
                rl_data[:, 6],
            )
            raceline = CubicSplineND(xs, ys, psis, kappas, vxs, axs)
        else:
            raceline = None

        return Track(
            xs=cl_xs,
            ys=cl_ys,
            centerline=centerline,
            raceline=raceline,
            occ_map=occ_map,
            resolution=resolution,
            ox=ox,
            oy=oy,
            oyaw=oyaw,
        )

    @staticmethod
    def from_numpy(waypoints: np.ndarray, s_frame_max, downsample_step=1):
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

        ss = waypoints[::downsample_step, 0]
        xs = waypoints[::downsample_step, 1]
        ys = waypoints[::downsample_step, 2]
        yaws = waypoints[::downsample_step, 3]
        ks = waypoints[::downsample_step, 4]
        vxs = waypoints[::downsample_step, 5]
        axs = waypoints[::downsample_step, 6]

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
            s_frame_max=s_frame_max,
        )

    @staticmethod
    def from_raceline_file(
        filepath: pathlib.Path, delimiter: str = ";", downsample_step=1
    ):
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

        ss = waypoints[::downsample_step, 0]
        xs = waypoints[::downsample_step, 1]
        ys = waypoints[::downsample_step, 2]
        yaws = waypoints[::downsample_step, 3]
        ks = waypoints[::downsample_step, 4]
        vxs = waypoints[::downsample_step, 5]
        axs = waypoints[::downsample_step, 6]

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
    def vmap_frenet_to_cartesian_jax(self, poses):
        s, ey, ephi = poses[:, 0], poses[:, 1], poses[:, 2]
        return jnp.asarray(
            jax.vmap(self.frenet_to_cartesian_jax, in_axes=(0, 0, 0))(s, ey, ephi)
        ).T

    @partial(jax.jit, static_argnums=(0))
    def frenet_to_cartesian_jax(self, s, ey, ephi):
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
        x, y = self.centerline.calc_position_jax(s)
        psi = self.centerline.calc_yaw_jax(s)

        # Adjust x,y by shifting along the normal vector
        x -= ey * jnp.sin(psi)
        y += ey * jnp.cos(psi)

        # Adjust psi by adding the heading deviation
        psi += ephi

        return x, y, jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    def cartesian_to_frenet(self, x, y, phi, s_guess=None):
        """
        Convert Cartesian coordinates to Frenet coordinates.

        x: x-coordinate
        y: y-coordinate
        phi: yaw angle

        returns:
            s: distance along the centerline
            ey: lateral deviation
            ephi: heading deviation
        """
        if s_guess is None:  # Utilize internal state to keep track of the guess
            s_guess = self.s_guess

        s, ey = self.centerline.calc_arclength(x, y, s_guess)
        # Wrap around
        s = s % self.s_frame_max

        self.s_guess = s  # Update the guess for the next iteration

        # Use the normal to calculate the signed lateral deviation
        # normal = self.centerline._calc_normal(s)
        yaw = self.centerline.calc_yaw(s)
        normal = np.asarray([-np.sin(yaw), np.cos(yaw)])
        x_eval, y_eval = self.centerline.calc_position(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = np.sign(np.dot([dx, dy], normal))
        ey = ey * distance_sign

        phi = phi - yaw
        return s, ey, np.arctan2(np.sin(phi), np.cos(phi))

    @partial(jax.jit, static_argnums=(0))
    def cartesian_to_frenet_jax(self, x, y, phi, s_guess=None):
        s, ey = self.centerline.calc_arclength_jax(x, y, s_guess)
        # Wrap around
        s = s % self.s_frame_max

        # Use the normal to calculate the signed lateral deviation
        # normal = self.centerline._calc_normal(s)
        yaw = self.centerline.calc_yaw_jax(s)
        normal = jnp.asarray([-jnp.sin(yaw), jnp.cos(yaw)])
        x_eval, y_eval = self.centerline.calc_position_jax(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = jnp.sign(jnp.dot(jnp.asarray([dx, dy]), normal))
        ey = ey * distance_sign

        phi = phi - yaw

        return s, ey, jnp.arctan2(jnp.sin(phi), jnp.cos(phi))

    @partial(jax.jit, static_argnums=(0))
    def vmap_cartesian_to_frenet_jax(self, poses):
        x, y, phi = poses[:, 0], poses[:, 1], poses[:, 2]
        return jnp.asarray(
            jax.vmap(self.cartesian_to_frenet_jax, in_axes=(0, 0, 0))(x, y, phi)
        ).T

    def curvature(self, s):
        """
        Get the curvature at a given s.

        s: distance along the raceline

        returns:
            curvature
        """
        s = s % self.s_frame_max
        return self.centerline.calc_curvature(s)

    @partial(jax.jit, static_argnums=(0))
    def curvature_jax(self, s):
        s = s % self.s_frame_max
        return self.centerline.calc_curvature_jax(s)
        # return self.centerline.find_curvature_jax(s)
