"""
Cubic Spline interpolation using scipy.interpolate
Provides utilities for position, curvature, yaw, and arclength calculation
"""

import numpy as np
import scipy.optimize as so
from scipy import interpolate
from typing import Optional
from functools import partial
import jax.numpy as jnp
import jax
from jax import lax


@jax.jit
def nearest_point_on_trajectory_jax(point, trajectory) -> tuple:
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = jnp.sum((point - trajectory[:-1, :]) * diffs[:, :], axis=1)
    t = jnp.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = jnp.linalg.norm(point - projections, axis=1)
    min_dist_segment = jnp.argmin(dists)
    return (
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@jax.jit
def first_point_on_trajectory(
    point: jnp.ndarray, trajectory: jnp.ndarray, radius: float
):
    # compute squared distances
    delta = trajectory - point  # shape (N,2)
    sq_dists = jnp.sum(delta**2, axis=-1)  # shape (N,)

    # mask out everything closer than radius
    sq_radius = radius**2
    masked_sq = jnp.where(sq_dists >= sq_radius, sq_dists, jnp.inf)

    # pick the minimal squared‐distance among the allowed set
    idx = jnp.argmin(masked_sq)

    closest_point = trajectory[idx]
    dist = jnp.sqrt(masked_sq[idx])

    return idx, closest_point, dist

@jax.jit
def _calc_yaw_from_xy(x, y):
    dx_dt = jnp.gradient(x)
    dy_dt = jnp.gradient(y)
    heading = jnp.arctan2(dy_dt, dx_dt)
    return heading

class CubicSplineND:
    """
    Cubic CubicSplineND class.

    Attributes
    ----------
    s : list
        cumulative distance along the data points.
    xs : np.ndarray
        x coordinates for data points.
    ys : np.ndarray
        y coordinates for data points.
    spsi: np.ndarray
        yaw angles for data points.
    ks : np.ndarray
        curvature for data points.
    vxs : np.ndarray
        velocity for data points.
    axs : np.ndarray
        acceleration for data points.
    """

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        psis: Optional[np.ndarray] = None,
        ks: Optional[np.ndarray] = None,
        vxs: Optional[np.ndarray] = None,
        axs: Optional[np.ndarray] = None,
    ):
        self.xs = xs
        self.ys = ys
        self.psis = psis  # Lets us know if yaw was provided
        self.ks = ks  # Lets us know if curvature was provided
        self.vxs = vxs  # Lets us know if velocity was provided
        self.axs = axs  # Lets us know if acceleration was provided

        psis_spline = psis if psis is not None else np.zeros_like(xs)
        # If yaw is provided, interpolate cosines and sines of yaw for continuity
        cosines_spline = np.cos(psis_spline)
        sines_spline = np.sin(psis_spline)

        ks_spline = ks if ks is not None else np.zeros_like(xs)
        vxs_spline = vxs if vxs is not None else np.zeros_like(xs)
        axs_spline = axs if axs is not None else np.zeros_like(xs)

        self.points = np.c_[
            self.xs,
            self.ys,
            cosines_spline,
            sines_spline,
            ks_spline,
            vxs_spline,
            axs_spline,
        ]
        self.points_jax = jnp.array(self.points)
        if not np.all(self.points[-1] == self.points[0]):
            self.points = np.vstack(
                (self.points, self.points[0])
            )  # Ensure the path is closed
        self.s = self.__calc_s(self.points[:, 0], self.points[:, 1])
        self.s_jax = jnp.array(self.s)
        # Use scipy CubicSpline to interpolate the points with periodic boundary conditions
        # This is necesaxsry to ensure the path is continuous
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type="periodic")
        self.spline_x_jax = jnp.array(self.spline.x)
        self.spline_c_jax = jnp.array(self.spline.c)

    def predict_with_spline(self, point, segment, state_index=0):
        # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
        # exp_x = (point - self.spline.x[[segment]])[None, :] ** np.arange(4)[::-1, None]
        exp_x = ((point - self.spline.x[segment]) ** np.arange(4)[::-1])[:, None]
        vec = self.spline.c[:, segment, state_index]
        # Sum over the rows of exp_x weighted by coefficients in the ith column of s.c
        point = vec.dot(exp_x)
        return np.asarray(point)

    @partial(jax.jit, static_argnums=(0))
    def predict_with_spline_jax(self, point, segment, state_index=0):
        # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
        exp_x = ((point - self.spline_x_jax[segment]) ** jnp.arange(4)[::-1])[:, None]
        vec = self.spline_c_jax[:, segment, state_index]
        # # Sum over the rows of exp_x weighted by coefficients in the ith column of s.c
        point = vec.dot(exp_x)
        return point

    def find_segment_for_x(self, x):
        # Find the segment of the spline that x is in
        return (x / self.spline.x[-1] * (len(self.spline_x_jax) - 2)).astype(int)
        # print(np.searchsorted(self.spline.x, x, side='right') - 1)
        # return np.searchsorted(self.spline.x, x, side='right') - 1

    @partial(jax.jit, static_argnums=(0))
    def find_segment_for_x_jax(self, x):
        # Find the segment of the spline that x is in
        # print((x / self.spline_x_jax[-1] * (len(self.spline_x_jax) - 2)).astype(int))
        return (x / self.spline_x_jax[-1] * (len(self.spline_x_jax) - 2)).astype(int)
        # return jnp.searchsorted(self.spline_x_jax, x, side='right') - 1

    def __calc_s(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calc cumulative distance.

        Parameters
        ----------
        x : list
            x coordinates for data points.
        y : list
            y coordinates for data points.

        Returns
        -------
        s : np.ndarray
            cumulative distance along the data points.
        """
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return np.array(s)

    def calc_position(self, s: float) -> np.ndarray:
        """
        Calc position at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float | None
            x position for given s.
        y : float | None
            y position for given s.
        """
        segment = self.find_segment_for_x(s)
        x = self.predict_with_spline(s, segment, 0)[0]
        y = self.predict_with_spline(s, segment, 1)[0]
        return x, y

    @partial(jax.jit, static_argnums=(0))
    def calc_position_jax(self, s: float) -> np.ndarray:
        """
        Calc position at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float | None
            x position for given s.
        y : float | None
            y position for given s.
        """
        segment = self.find_segment_for_x_jax(s)
        x = self.predict_with_spline_jax(s, segment, 0)[0]
        y = self.predict_with_spline_jax(s, segment, 1)[0]
        return x, y

    def calc_curvature(self, s: float) -> Optional[float]:
        """
        Calc curvature at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        if self.ks is None:  # curvature was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            ddx, ddy = self.spline(s, 2)[:2]
            k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
            return k
        else:
            segment = self.find_segment_for_x(s)
            k = self.predict_with_spline(s, segment, 4)[0]
            return k

    def calc_curvature_jax(self, s: float) -> Optional[float]:
        """
        Calc curvature at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        if self.ks is None:  # curvature was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            ddx, ddy = self.spline(s, 2)[:2]
            k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
            return k
        else:
            segment = self.find_segment_for_x_jax(s)
            k = self.predict_with_spline_jax(s, segment, 4)[0]
            return k

    def find_curvature_jax(self, s: float) -> Optional[float]:
        segment = self.find_segment_for_x_jax(s)
        k = self.points_jax[segment, 4]
        return k

    def find_curvature(self, s: float) -> Optional[float]:
        """
        Find curvature at the given s by the segment.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        if self.ks is None:  # curvature was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            ddx, ddy = self.spline(s, 2)[:2]
            k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
            return k
        else:
            segment = self.find_segment_for_x(s)
            k = self.points[segment, 4]
            return k

    def calc_yaw(self, s: float) -> Optional[float]:
        """
        Calc yaw angle at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. If `s` is outside the data point's range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        if self.psis is None:  # yaw was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            yaw = np.arctan2(dy, dx)
            # Convert yaw to [0, 2pi]
            # yaw = (yaw + 2 * math.pi) % (2 * math.pi)
            return yaw
        else:
            segment = self.find_segment_for_x(s)
            cos = self.predict_with_spline(s, segment, 2)[0]
            sin = self.predict_with_spline(s, segment, 3)[0]
            # yaw = (math.atan2(sin, cos) + 2 * math.pi) % (2 * math.pi)
            yaw = np.arctan2(sin, cos)
            return yaw

    @partial(jax.jit, static_argnums=(0))
    def calc_yaw_jax(self, s: float) -> Optional[float]:
        segment = self.find_segment_for_x_jax(s)
        cos = self.predict_with_spline_jax(s, segment, 2)[0]
        sin = self.predict_with_spline_jax(s, segment, 3)[0]
        yaw = jnp.arctan2(sin, cos)
        return yaw

    def calc_arclength(self, x: float, y: float, s_guess=0.0) -> tuple[float, float]:
        """
        Fast calculation of arclength for a given point (x, y) on the trajectory.
        Less accuarate and less smooth than calc_arclength but much faster.
        Suitable for lap counting.

        Parameters
        ----------
        x : float
            x position.
        y : float
            y position.

        Returns
        -------
        s : float
            distance from the start point for given x, y.
        ey : float
            lateral deviation for given x, y.
        """
        ey, t, min_dist_segment = nearest_point_on_trajectory_jax(
            np.array([x, y]).astype(np.float32), self.points[:, :2]
        )
        # s = s at closest_point + t
        s = float(
            self.s[min_dist_segment]
            + t * (self.s[min_dist_segment + 1] - self.s[min_dist_segment])
        )

        return s, ey

    @partial(jax.jit, static_argnums=(0))
    def calc_arclength_jax(
        self, x: float, y: float, s_guess=0.0
    ) -> tuple[float, float]:
        ey, t, min_dist_segment = nearest_point_on_trajectory_jax(
            jnp.array([x, y]), self.points_jax[:, :2]
        )
        s = self.s_jax[min_dist_segment] + t * (
            self.s_jax[min_dist_segment + 1] - self.s_jax[min_dist_segment]
        )
        return s, ey

    def calc_arclength_slow(
        self, x: float, y: float, s_guess: float = 0.0
    ) -> tuple[float, float]:
        """
        Calculate arclength for a given point (x, y) on the trajectory.

        Parameters
        ----------
        x : float
            x position.
        y : float
            y position.
        s_guess : float
            initial guess for s.

        Returns
        -------
        s : float
            distance from the start point for given x, y.
        ey : float
            lateral deviation for given x, y.
        """

        def distance_to_spline(s):
            x_eval, y_eval = self.spline(s)[0]
            return np.sqrt((x - x_eval) ** 2 + (y - y_eval) ** 2)

        output = so.fmin(distance_to_spline, s_guess, full_output=True, disp=False)
        closest_s = float(output[0][0])
        absolute_distance = output[1]
        return closest_s, absolute_distance

    def _calc_tangent(self, s: float) -> np.ndarray:
        """
        Calculates the tangent to the curve at a given point.

        Parameters
        ----------
        s : float
            distance from the start point.
            If `s` is outside the data point's range, return None.

        Returns
        -------
        tangent : float
            tangent vector for given s.
        """
        dx, dy = self.spline(s, 1)[:2]
        tangent = np.array([dx, dy])
        return tangent

    def _calc_normal(self, s: float) -> np.ndarray:
        """
        Calculate the normal to the curve at a given point.

        Parameters
        ----------
        s : float
            distance from the start point.
            If `s` is outside the data point's range, return None.

        Returns
        -------
        normal : float
            normal vector for given s.
        """
        dx, dy = self.spline(s, 1)[:2]
        normal = np.array([-dy, dx])
        return normal

    def calc_velocity(self, s: float) -> Optional[float]:
        """
        Calc velocity at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        v : float
            velocity for given s.
        """
        if self.vxs is None:  # velocity was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            v = np.hypot(dx, dy)
            return v
        else:
            segment = self.find_segment_for_x(s)
            v = self.predict_with_spline(s, segment, 5)[0]
            return v

    def calc_acceleration(self, s: float) -> Optional[float]:
        """
        Calc acceleration at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        a : float
            acceleration for given s.
        """
        if self.axs is None:  # acceleration was not provided => numerical calculation
            ddx, ddy = self.spline(s, 2)[:2]
            a = np.hypot(ddx, ddy)
            return a
        else:
            segment = self.find_segment_for_x(s)
            a = self.predict_with_spline(s, segment, 6)[0]
            return a
