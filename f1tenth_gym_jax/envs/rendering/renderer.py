from __future__ import annotations
import logging
from typing import Any, Callable, Optional
import signal

import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6 import QtGui
import pyqtgraph as pg
from pyqtgraph.examples.utils import FrameCounter
from pyqtgraph.exporters import ImageExporter
from PIL import ImageColor

from .objects import (
    Car,
    TextObject,
)
from ..f110_env import F110Env

# one-line instructions visualized at the top of the screen (if show_info=True)
INSTRUCTION_TEXT = "Mouse click (L/M/R): Change POV - 'S' key: On/Off"


class TrajRenderer:
    """
    Renderer of the environment using PyQtGraph.
    """

    def __init__(
        self,
        env: F110Env,
        render_fps: int = 100,
        window_width: int = 800,
        window_height: int = 600,
    ):
        """
        Initialize the Pygame renderer.

        Parameters
        ----------
        params : dict
            dictionary of simulation parameters (including vehicle dimensions, etc.)
        track : Track
            track object
        agent_ids : list
            list of agent ids to render
        render_spec : RenderSpec
            rendering specification
        render_mode : str
            rendering mode in ["human", "human_fast", "rgb_array"]
        render_fps : int
            number of frames per second
        """
        self.env = env

        self.cars = None
        self.sim_time = None
        self.window = None
        self.canvas = None

        self.render_fps = render_fps

        # create the canvas
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.window = pg.GraphicsLayoutWidget()
        self.window.setWindowTitle("f1tenth_gym_jax - Trajectory Renderer")
        self.window.setGeometry(0, 0, window_width, window_height)
        self.canvas: pg.PlotItem = self.window.addPlot()

        # Disable interactivity
        self.canvas.setMouseEnabled(x=False, y=False)  # Disable mouse panning & zooming
        self.canvas.hideButtons()  # Disable corner auto-scale button
        self.canvas.setMenuEnabled(False)  # Disable right-click context menu

        legend = self.canvas.addLegend()  # This doesn't disable legend interaction
        # Override both methods responsible for mouse events
        legend.mouseDragEvent = lambda *args, **kwargs: None
        legend.hoverEvent = lambda *args, **kwargs: None
        # self.scene() is a pyqtgraph.GraphicsScene.GraphicsScene.GraphicsScene
        self.window.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.window.keyPressEvent = self.key_pressed

        # Remove axes
        self.canvas.hideAxis("bottom")
        self.canvas.hideAxis("left")

        # setting plot window background color
        self.window.setBackground("w")

        # widgets
        self.cw = QtWidgets.QWidget()
        self.cw.setLayout(QtWidgets.QGridLayout())

        # fps and time renderer
        self.clock = FrameCounter()
        self.fps_renderer = TextObject(parent=self.canvas, position="bottom_left")
        self.time_renderer = TextObject(parent=self.canvas, position="bottom_right")
        self.bottom_info_renderer = TextObject(
            parent=self.canvas, position="bottom_center"
        )
        self.top_info_renderer = TextObject(parent=self.canvas, position="top_center")

        # trajectory selector
        self.selector = pg.SpinBox(
            value=0,
            bounds=[0, self.num_trajectories - 1],
            int=True,
            minStep=1,
            step=1,
            wrapping=True,
        )

        if self.render_mode in ["human", "human_fast"]:
            self.clock.sigFpsUpdate.connect(
                lambda fps: self.fps_renderer.render(f"FPS: {fps:.1f}")
            )

        colors_rgb = [
            [rgb for rgb in ImageColor.getcolor(c, "RGB")]
            for c in render_spec.vehicle_palette
        ]
        self.car_colors = [
            colors_rgb[i % len(colors_rgb)] for i in range(len(self.agent_ids))
        ]

        width, height = render_spec.window_size, render_spec.window_size

        # map metadata
        self.map_origin = track.spec.origin
        self.map_resolution = track.spec.resolution

        # load map image
        original_img = track.occupancy_map

        # convert shape from (W, H) to (W, H, 3)
        track_map = np.stack([original_img, original_img, original_img], axis=-1)

        # rotate and flip to match the track orientation
        track_map = np.rot90(track_map, k=1)  # rotate clockwise
        track_map = np.flip(track_map, axis=0)  # flip vertically

        self.image_item = pg.ImageItem(track_map)
        # Example: Transformed display of ImageItem
        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        # Translate image by the origin of the map
        tr.translate(self.map_origin[0], self.map_origin[1])
        # Scale image by the resolution of the map
        tr.scale(self.map_resolution, self.map_resolution)
        self.image_item.setTransform(tr)
        self.canvas.addItem(self.image_item)

        # callbacks for custom visualization, called at every rendering step
        self.callbacks = []

        # event handling flags
        self.draw_flag: bool = True
        if render_spec.focus_on:
            self.active_map_renderer = "car"
            self.follow_agent_flag: bool = True
            self.agent_to_follow: int = self.agent_ids.index(render_spec.focus_on)
        else:
            self.active_map_renderer = "map"
            self.follow_agent_flag: bool = False
            self.agent_to_follow: int = None

        if self.render_mode in ["human", "human_fast"]:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            self.window.show()
        elif self.render_mode == "rgb_array":
            self.exporter = ImageExporter(self.canvas)

    def update(self, state: dict) -> None:
        """
        Update the simulation state to be rendered.

        Parameters
        ----------
            state: simulation state as dictionary
        """
        if self.cars is None:
            self.cars = [
                Car(
                    car_length=self.params["length"],
                    car_width=self.params["width"],
                    color=self.car_colors[ic],
                    render_spec=self.render_spec,
                    map_origin=self.map_origin[:2],
                    resolution=self.map_resolution,
                    parent=self.canvas,
                )
                for ic in range(len(self.agent_ids))
            ]

        # update cars state and zoom level (updating points-per-unit)
        for i in range(len(self.agent_ids)):
            self.cars[i].update(state, i)

        # update time
        self.sim_time = state["sim_time"]

    def add_renderer_callback(self, callback_fn: Callable[[EnvRenderer], None]) -> None:
        """
        Add a custom callback for visualization.
        All the callbacks are called at every rendering step, after having rendered the map and the cars.

        Parameters
        ----------
        callback_fn : Callable[[EnvRenderer], None]
            callback function to be called at every rendering step
        """
        self.callbacks.append(callback_fn)

    def key_pressed(self, event: QtGui.QKeyEvent) -> None:
        """
        Handle key press events.

        Parameters
        ----------
        event : QtGui.QKeyEvent
            key event
        """
        if event.key() == QtCore.Qt.Key.Key_S:
            logging.debug("Pressed S key -> Enable/disable rendering")
            self.draw_flag = not self.draw_flag
            self.draw_flag_changed = True

    def mouse_clicked(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse click events.

        Parameters
        ----------
        event : QtGui.QMouseEvent
            mouse event
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            logging.debug("Pressed left button -> Follow Next agent")

            self.follow_agent_flag = True
            if self.agent_to_follow is None:
                self.agent_to_follow = 0
            else:
                self.agent_to_follow = (self.agent_to_follow + 1) % len(self.agent_ids)

            self.active_map_renderer = "car"
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            logging.debug("Pressed right button -> Follow Previous agent")

            self.follow_agent_flag = True
            if self.agent_to_follow is None:
                self.agent_to_follow = 0
            else:
                self.agent_to_follow = (self.agent_to_follow - 1) % len(self.agent_ids)

            self.active_map_renderer = "car"
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            logging.debug("Pressed middle button -> Change to Map View")

            self.follow_agent_flag = False
            self.agent_to_follow = None

            self.active_map_renderer = "map"

    def render(self, trajectory: np.ndarray) -> Optional[np.ndarray]:
        """
        Render the current state in a frame.
        It renders in the order: map, cars, callbacks, info text.

        Returns
        -------
        Optional[np.ndarray]
            if render_mode is "rgb_array", returns the rendered frame as an array
        """
        # get sizes
        num_steps, num_envs, num_agents, num_states = trajectory.shape

        # draw cars
        for i in range(len(self.agent_ids)):
            self.cars[i].render()

        # call callbacks
        for callback_fn in self.callbacks:
            callback_fn(self)

        if self.follow_agent_flag:
            ego_x, ego_y = self.cars[self.agent_to_follow].pose[:2]
            self.canvas.setXRange(ego_x - 10, ego_x + 10)
            self.canvas.setYRange(ego_y - 10, ego_y + 10)
        else:
            self.canvas.autoRange()

        agent_to_follow_id = (
            self.agent_ids[self.agent_to_follow]
            if self.agent_to_follow is not None
            else None
        )
        self.bottom_info_renderer.render(text=f"Focus on: {agent_to_follow_id}")

        if self.render_spec.show_info:
            self.top_info_renderer.render(text=INSTRUCTION_TEXT)

        self.time_renderer.render(text=f"{self.sim_time:.2f}")
        self.clock.update()
        self.app.processEvents()

        if self.render_mode in ["human", "human_fast"]:
            assert self.window is not None

        else:
            # rgb_array mode => extract the frame from the canvas
            qImage = self.exporter.export(toBytes=True)

            width = qImage.width()
            height = qImage.height()

            ptr = qImage.bits()
            ptr.setsize(height * width * 4)
            frame = np.array(ptr).reshape(height, width, 4)  #  Copies the data

            return frame[:, :, :3]  # remove alpha channel

    def render_points(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> pg.PlotDataItem:
        """
        Render a sequence of xy points on screen.

        Parameters
        ----------
        points : list | np.ndarray
            list of points to render
        color : Optional[tuple[int, int, int]], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : Optional[int], optional
            size of the points in pixels, by default 1
        """
        return self.canvas.plot(
            points[:, 0],
            points[:, 1],
            pen=None,
            symbol="o",
            symbolPen=pg.mkPen(color=color, width=0),
            symbolBrush=pg.mkBrush(color=color, width=0),
            symbolSize=size,
        )

    def render_lines(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> pg.PlotDataItem:
        """
        Render a sequence of lines segments.

        Parameters
        ----------
        points : list | np.ndarray
            list of points to render
        color : Optional[tuple[int, int, int]], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : Optional[int], optional
            size of the line, by default 1
        """
        pen = pg.mkPen(color=pg.mkColor(*color), width=size)
        return self.canvas.plot(
            points[:, 0], points[:, 1], pen=pen, fillLevel=None, antialias=True
        )  ## setting pen=None disables line drawing

    def render_closed_lines(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> pg.PlotDataItem:
        """
        Render a sequence of lines segments forming a closed loop (draw a line between the last and the first point).

        Parameters
        ----------
        points : list | np.ndarray
            list of 2d points to render
        color : Optional[tuple[int, int, int]], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : Optional[int], optional
            size of the line, by default 1
        """
        # Append the first point to the end to close the loop
        points = np.vstack([points, points[0]])

        pen = pg.mkPen(color=pg.mkColor(*color), width=size)
        pen.setCapStyle(pg.QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(pg.QtCore.Qt.PenJoinStyle.RoundJoin)

        return self.canvas.plot(
            points[:, 0], points[:, 1], pen=pen, cosmetic=True, antialias=True
        )  ## setting pen=None disables line drawing

    def close(self) -> None:
        """
        Close the rendering environment.
        """
        self.app.exit()
