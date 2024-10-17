"""
Cubic Spline interpolation using scipy.interpolate
Provides utilities for position, curvature, yaw, and arclength calculation
"""

import numpy as np
from scipy import interpolate
from typing import Optional
from functools import partial
import jax.numpy as jnp
import jax


@partial(jax.jit, static_argnums=(1))
def nearest_point_on_trajectory(point, trajectory) -> tuple:
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
        self.points = jnp.array(self.points)
        if not np.all(self.points[-1] == self.points[0]):
            self.points = np.vstack(
                (self.points, self.points[0])
            )  # Ensure the path is closed
        self.s = self.__calc_s(self.points[:, 0], self.points[:, 1])
        self.s = jnp.array(self.s)
        # Use scipy CubicSpline to interpolate the points with periodic boundary conditions
        # This is necesaxsry to ensure the path is continuous
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type="periodic")
        self.spline_x = jnp.array(self.spline.x)
        self.spline_c = jnp.array(self.spline.c)

    @partial(jax.jit, static_argnums=(0))
    def predict_with_spline(self, point, segment, state_index=0):
        # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
        exp_x = ((point - self.spline_x[segment]) ** jnp.arange(4)[::-1])[:, None]
        vec = self.spline_c[:, segment, state_index]
        # # Sum over the rows of exp_x weighted by coefficients in the ith column of s.c
        point = vec.dot(exp_x)
        return point

    @partial(jax.jit, static_argnums=(0))
    def find_segment_for_x(self, x):
        # Find the segment of the spline that x is in
        # print((x / self.spline_x[-1] * (len(self.spline_x) - 2)).astype(int))
        return (x / self.spline_x[-1] * (len(self.spline_x) - 2)).astype(int)
        # return jnp.searchsorted(self.spline_x, x, side='right') - 1

    @partial(jax.jit, static_argnums=(0))
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

    @partial(jax.jit, static_argnums=(0))
    def find_curvature(self, s: float) -> Optional[float]:
        segment = self.find_segment_for_x(s)
        k = self.points[segment, 4]
        return k

    @partial(jax.jit, static_argnums=(0))
    def calc_yaw(self, s: float) -> Optional[float]:
        if self.psis is None:  # yaw was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            yaw = jnp.arctan2(dy, dx)
            # Convert yaw to [0, 2pi]
            # yaw = yaw % (2 * jnp.pi)
            return yaw
        else:
            segment = self.find_segment_for_x(s)
            cos = self.predict_with_spline(s, segment, 2)[0]
            sin = self.predict_with_spline(s, segment, 3)[0]
            yaw = jnp.arctan2(sin, cos)
            # yaw = (jnp.arctan2(sin, cos) + 2 * jnp.pi) % (2 * jnp.pi) # Get yaw from cos,sin and convert to [0, 2pi]
            return yaw

    @partial(jax.jit, static_argnums=(0))
    def calc_arclength(self, x: float, y: float, s_guess=0.0) -> tuple[float, float]:
        ey, t, min_dist_segment = nearest_point_on_trajectory(
            jnp.array([x, y]), self.points[:, :2]
        )
        s = self.s[min_dist_segment] + t * (
            self.s[min_dist_segment + 1] - self.s[min_dist_segment]
        )
        return s, ey
