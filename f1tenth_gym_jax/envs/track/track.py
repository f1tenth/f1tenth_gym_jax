from __future__ import annotations
import pathlib
from typing import Optional
import numpy as np
from functools import partial
import jax.numpy as jnp
import jax
from numba import njit

from .cubic_spline import CubicSplineND, CubicSpline2D


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
        return jnp.asarray(jax.vmap(self.frenet_to_cartesian_jax, in_axes=(0, 0, 0))(
            s, ey, ephi
            )).T
    
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
        if s_guess is None: # Utilize internal state to keep track of the guess
            s_guess = self.s_guess

        s, ey = self.centerline.calc_arclength(x, y, s_guess)
        # Wrap around
        s = s % self.s_frame_max

        self.s_guess = s # Update the guess for the next iteration

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
        return jnp.asarray(jax.vmap(self.cartesian_to_frenet_jax, in_axes=(0, 0, 0))(
            x, y, phi
            )).T

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

    def centerline_to_frenet(trajectory, velocity=5.0):   
        '''
        Converts a trajectory in the form [x_m, y_m, w_tr_right_m, w_tr_left_m] to [s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2, w_tr_right_m, w_tr_left_m]
        Assumes constant velocity

        Parameters
        ----------
        trajectory : np.array
            Trajectory in the form [x_m, y_m, w_tr_right_m, w_tr_left_m]
        velocity : float, optional
            Velocity of the vehicle, by default 5.0
        '''
        # Initialize variables
        # eps = 1e-5 * (np.random.randint(0, 1) - 1)
        eps = 0
        s = 0.0
        x = trajectory[0, 0]
        y = trajectory[0, 1]
        psi = np.arctan2((trajectory[1, 1] - trajectory[0, 1]), (trajectory[1, 0] - trajectory[0, 0])) 
        # curvature = calculate_curvature(trajectory[:, 0], trajectory[:, 1])
        waypoint_spline = CubicSpline2D(trajectory[:, 0], trajectory[:, 1])
        vx = velocity
        ax = 0.0
        width_l = trajectory[0, 2]
        width_r = trajectory[0, 3]

        # Initialize output
        output = np.zeros((trajectory.shape[0], 7))
        output[0, :] = np.array([s, x, y, psi, waypoint_spline.calc_curvature(s), vx, ax])

        # Iterate over trajectory
        for i in range(1, trajectory.shape[0]):
            # Calculate s
            s += np.sqrt((trajectory[i, 0] - trajectory[i-1, 0])**2 + (trajectory[i, 1] - trajectory[i-1, 1])**2)
            # Calculate psi
            psi = np.arctan2((trajectory[i, 1] - trajectory[i-1, 1]), (trajectory[i, 0] - trajectory[i-1, 0]))
            # Calculate kappa
            # eps = 1e-5 * (np.random.randint(0, 1) - 1)
            eps = 0
            kappa = waypoint_spline.calc_curvature(s)
            if kappa < 5e-3: kappa = 0
            # kappa = 0
            # kappa = (trajectory[i, 3] - trajectory[i, 2]) / (2 * np.sqrt((trajectory[i, 0] - trajectory[i-1, 0])**2 + (trajectory[i, 1] - trajectory[i-1, 1])**2)) + eps
            # Calculate ax
            ax = 0.0

            # Save to output
            output[i, :] = np.array([s, trajectory[i, 0], trajectory[i, 1], psi, kappa, vx, ax])

        return output
    
    def get_refernece_traj(self, state, target_speed=None, vind=5, speed_factor=1.0,
                           horizon=10, DT=0.1):
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                            self.waypoints[:, (1, 2)].copy())
        
        if target_speed is None:
            # speed = self.waypoints[ind, vind] * speed_factor
            # speed = np.minimum(self.waypoints[ind, vind] * speed_factor, 20.)
            speed = state[3]
        else:
            speed = target_speed
        
        # if ind < self.waypoints.shape[0] - self.n_steps:
        #     speeds = self.waypoints[ind:ind+self.n_steps, vind]
        # else:
        speeds = np.ones(horizon) * speed
        
        reference = get_reference_trajectory(speeds, dist, ind, 
                                            self.waypoints.copy(), int(horizon),
                                            self.waypoints_distances.copy(), DT=DT)
        orientation = state[4]
        reference[3, :][reference[3, :] - orientation > 5] = np.abs(
            reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -5] = np.abs(
            reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))
        
        # reference[2] = np.where(reference[2] - speed > 5.0, speed + 5.0, reference[2])
        self.reference = reference.T
        return reference.T, ind
    
if __name__ == "__main__":
    # Load the racline.csv
    # track = Track.from_raceline_file(
    #     filepath=pathlib.Path("raceline.csv"), delimiter=";", downsample_step=10
    # )
    # waypoints = np.hstack((track.raceline.s.reshape(-1,1), track.raceline.points))
    
    from main import Config
    
    config = Config()
    config.map_ind = 58
    map_info = np.genfromtxt(config.map_dir + 'map_info.txt', delimiter='|', dtype='str')
    track, config = Track.load_map(config.map_dir, map_info, config.map_ind, config, scale=config.map_scale, downsample_step=1)

    print(track.frenet_to_cartesian(369.5279, -0.33358, 0.05985))
    print(track.frenet_to_cartesian_jax(369.5279, -0.33358, 0.05985))
    
    print(track.frenet_to_cartesian(0, 0.33358, 0.05985))
    print(track.frenet_to_cartesian_jax(0, 0.33358, 0.05985))
    

    # Get a random s, ey point and plot it colored by curvature
    POINTS_TO_TEST = 1000
    s = np.random.uniform(0, track.raceline.s[-1], size=(POINTS_TO_TEST))
    ey = np.random.uniform(-2, 2, size=(POINTS_TO_TEST))
    ephi = 0
    points = np.array([s, ey]).T
    X, Y, CURV = [], [], []
    VXS, AXS = [], []
    for point in points:
        s_, ey_ = point
        x, y, psi = track.frenet_to_cartesian(s_, ey_, ephi)
        curv = track.curvature(s_)
        # print(f"s: {s}, ey: {ey}, ephi: {ephi}, x: {x}, y: {y}, psi: {psi}, curv: {curv}")
        X.append(x)
        Y.append(y)
        CURV.append(curv)
        vxs = track.raceline.calc_velocity(s_)
        axs = track.raceline.calc_acceleration(s_)
        VXS.append(vxs)
        AXS.append(axs)

    x, y = np.array(X), np.array(Y)
    CURV = np.array(CURV)
    VXS, AXS = np.array(VXS), np.array(AXS)

    # Scatter raceline xs, ys colored by curvature
    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(track.raceline.xs, track.raceline.ys, c=track.raceline.ks)
    # bar = plt.colorbar(label="Curvature", extend='both')
    # plt.scatter(x, y, c=CURV, edgecolors='black', vmin=bar.get_ticks().min(), vmax=bar.get_ticks().max())
    # plt.title("Curvature Sampled vs Raceline")
    
    # plt.figure()
    # plt.scatter(track.raceline.xs, track.raceline.ys, c=track.raceline.vxs)
    # bar = plt.colorbar(label="Velocity", extend='both')
    # plt.scatter(x, y, c=VXS, edgecolors='black', vmin=bar.get_ticks().min(), vmax=bar.get_ticks().max())
    # plt.title("Velocity Sampled vs Raceline")

    # fig = plt.figure()
    # plt.scatter(track.raceline.xs, track.raceline.ys, c=track.raceline.axs)
    # bar = plt.colorbar(label="Acceleration", extend='both')
    # plt.scatter(x, y, c=AXS, edgecolors='black', vmin=bar.get_ticks().min(), vmax=bar.get_ticks().max())
    # plt.title("Acceleration Sampled vs Raceline")
    # plt.show()

    # Test out frenet to cartesian and cartesian to frenet
    POINTS_TO_TEST = 3000
    s = np.random.uniform(0, track.raceline.s[-1], size=(POINTS_TO_TEST))
    ey = np.random.uniform(-2, 2, size=(POINTS_TO_TEST))
    ephi = np.random.uniform(-np.pi, np.pi, size=(POINTS_TO_TEST))
    points = []
    # for i in range(POINTS_TO_TEST):
    #     x, y, psi = track.frenet_to_cartesian(s[i], ey[i], ephi[i])
    #     points.append([x, y, psi])
    #     s_, ey_, ephi_ = track.cartesian_to_frenet(x, y, psi)
    #     if np.abs(s[i] - s_) > 0.01: print(f"Expected s: {s[i]}, got: {s_}")
    #     if np.abs(ey[i] - ey_) > 0.01: print(f"Expected ey: {ey[i]}, got: {ey_}")
    #     if np.abs(ephi[i] - ephi_) > 0.01: print(f"Expected ephi: {ephi[i]}, got: {ephi_}")
    # points = np.array(points)
    # print("Simple Frenet to Cartesian and Cartesian to Frenet tests passed!")
    print(track.s_frame_max)
    pose_carte = track.vmap_frenet_to_cartesian_jax(jnp.vstack([s, ey, ephi]).T)
    points = track.vmap_cartesian_to_frenet_jax(pose_carte)
    print(points.shape)
    # for i in range(POINTS_TO_TEST):
    #     x, y, psi = track.frenet_to_cartesian(s[i], ey[i], ephi[i])
    #     x1, y1, psi1 = track.frenet_to_cartesian_jax(s[i], ey[i], ephi[i])
    #     if np.abs(x - x1) > 0.01: print(f"Expected s: {x}, got: {x1}")
    #     if np.abs(y - y1) > 0.01: print(f"Expected ey: {y}, got: {y1}")
    #     if np.abs(psi - psi1) > 0.01: print(f"Expected ephi: {psi}, got: {psi1}")
        
    #     s_, ey_, ephi_ = track.cartesian_to_frenet(x, y, psi)
    #     s_1, ey_1, ephi_1 = track.cartesian_to_frenet_jax(x1, y1, psi1)
    #     if np.abs(s_ - s_1) > 0.01: print(f"Expected s: {s_}, got: {s_1}")
    #     if np.abs(ey_ - ey_1) > 0.01: 
    #         print(f"Expected ey: {ey[i]}")
    #         print(f"Expected ey: {ey_}, got: {ey_1}")
    #         print(f"Expected s: {s_}, got: {s_1}")
    #         print(f"Expected ephi: {ephi_}, got: {ephi_1}")
    #     if np.abs(ephi_ - ephi_1) > 0.01: print(f"Expected ephi: {ephi_}, got: {ephi_1}")
    
    
    for i in range(POINTS_TO_TEST):
        if np.abs(s[i] - points[i, 0]) > 0.01: print(f"Expected x: {s[i]}, got: {points[i, 0]}")
        if np.abs(ey[i] - points[i, 1]) > 0.01: print(f"Expected y: {ey[i]}, got: {points[i, 1]}")
        if np.abs(ephi[i] - points[i, 2]) > 0.01: print(f"Expected psi: {ephi[i]}, got: {points[i, 2]}")
    print("JAX Simple Frenet to Cartesian and Cartesian to Frenet tests passed!")
    
    fig = plt.figure()
    plt.scatter(track.raceline.xs, track.raceline.ys, c=track.raceline.axs)
    bar = plt.colorbar(label="Position", extend='both')
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], edgecolors='black', vmin=bar.get_ticks().min(), vmax=bar.get_ticks().max())
    plt.title("Position Sampled vs Raceline")
    plt.show()
    
    
    
    # Test out frenet to cartesian and cartesian to frenet
    # x = 0
    # y = 0
    # psi = 0
    # s, ey, ephi = track.cartesian_to_frenet(x, y, psi)
    # x_, y_, psi_ = track.frenet_to_cartesian(s, ey, ephi)
    # assert np.isclose(x, x_), f"Expected x: {x}, got: {x_}"
    # assert np.isclose(y, y_), f"Expected y: {y}, got: {y_}"
    # assert np.isclose(psi, psi_), f"Expected psi: {psi}, got: {psi_}"
    # print("Simple Cartesian to Frenet and Frenet to Cartesian tests passed!")


@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment
    
def get_reference_trajectory(predicted_speeds, dist_from_segment_start, idx, 
                             waypoints, n_steps, waypoints_distances, DT):
    s_relative = np.zeros((n_steps + 1,))
    s_relative[0] = dist_from_segment_start
    s_relative[1:] = predicted_speeds * DT
    s_relative = np.cumsum(s_relative)

    waypoints_distances_relative = np.cumsum(np.roll(waypoints_distances, -idx))

    index_relative = np.int_(np.ones((n_steps + 1,)))
    for i in range(n_steps + 1):
        index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
    index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

    segment_part = s_relative - (
            waypoints_distances_relative[index_relative] - waypoints_distances[index_absolute])

    t = (segment_part / waypoints_distances[index_absolute])
    # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

    position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                        waypoints[index_absolute][:, (1, 2)])
    position_diff_s = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 0] -
                        waypoints[index_absolute][:, 0])
    orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                            waypoints[index_absolute][:, 3])
    speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                    waypoints[index_absolute][:, 5])

    interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T
    interpolated_s = waypoints[index_absolute][:, 0] + (t * position_diff_s)
    interpolated_s[np.where(interpolated_s > waypoints[-1, 0])] -= waypoints[-1, 0]
    interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
    interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi
    interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)
    
    reference = np.array([
        # Sort reference trajectory so the order of reference match the order of the states
        interpolated_positions[:, 0],
        interpolated_positions[:, 1],
        interpolated_speeds,
        interpolated_orientations,
        # Fill zeros to the rest so number of references mathc number of states (x[k] - ref[k])
        interpolated_s,
        np.zeros(len(interpolated_speeds)),
        np.zeros(len(interpolated_speeds))
    ])
    return reference

def points_in_convex_hull(points_hull, points):
    """
    # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    # def point_in_hull(point, hull, tolerance=1e-12):
    #     return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance)
    #         for eq in hull.equations)
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points_hull)
    eq = np.asarray(hull.equations)
    tolerance=1e-2
    arr = np.all(np.sum(eq[:, :-1][:, None, :].repeat(points.shape[0], axis=1) * \
        points[None, :, :].repeat(eq.shape[0], axis=0), axis=2) + \
            eq[:, -1][:, None].repeat(points.shape[0], axis=1) <= tolerance, axis=0)
    return arr