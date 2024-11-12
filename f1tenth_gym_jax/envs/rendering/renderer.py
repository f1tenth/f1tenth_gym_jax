from __future__ import annotations
import logging
from typing import Any, Callable, Optional

import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6 import QtGui
import pyqtgraph as pg
from pyqtgraph.examples.utils import FrameCounter
from pyqtgraph.exporters import ImageExporter
from PIL import ImageColor

from ..f110_env import F110Env
from ..collision_models import get_vertices
from .objects import _get_tire_vertices

class TrajRenderer:
    """
    Renderer of the environment using PyQtGraph.
    """

    def __init__(
        self,
        env: F110Env,
        render_fps: int = 100,
        window_width: int = 800,
        window_height: int = 800,
        render_mode: str = "human",
    ):
        """
        Initialize the Pygame renderer.

        Parameters
        ----------
        env : F110Env
            environment to render
        render_fps : int, optional
        window_width : int, optional
        window_height : int, optional
        render_mode : str, optional
            rendering mode, by default "human", choose from ["human", "human_fast", "rgb_array"]
        """
        self.env = env
        self.render_mode = render_mode

        self.num_trajectories = 1
        self.num_agents = env.num_agents

        self.cars = None
        self.sim_time = 0.0
        self.window = None
        self.canvas = None
        self.current_step = 0
        self.current_traj = 0

        self.render_fps = render_fps

        self.thickness = 1.0
        self.wheel_length = 0.2
        self.wheel_width = 0.1

        # create the canvas
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.window = pg.GraphicsLayoutWidget()
        self.window.setFixedSize(window_width, window_height + 50)

        self.window.setWindowTitle("f1tenth_gym_jax - Trajectory Renderer")
        self.layout = QtWidgets.QGridLayout()
        self.window.setLayout(self.layout)

        self.canvas = pg.PlotWidget()
        self.canvas.setFixedSize(window_width, window_height)
        self.canvas.setAspectLocked(True)
        self.canvas.getPlotItem().autoRange()

        self.layout.addWidget(
            self.canvas, 0, 0, 1, 6, QtCore.Qt.AlignmentFlag.AlignHCenter
        )

        # map
        # map metadata
        self.map_origin = [env.track.ox, env.track.oy, env.track.oyaw]
        self.map_resolution = env.track.resolution

        # load map image
        original_img = env.track.occ_map

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

        spin_text = QtWidgets.QLabel("Trajectory #:")
        self.layout.addWidget(spin_text, 1, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        # trajectory selector
        self.traj_selector = pg.SpinBox(
            value=0,
            bounds=[0, self.num_trajectories],
            int=True,
            minStep=1,
            step=1,
            wrapping=True,
        )
        self.traj_selector.setFixedSize(100, 20)
        self.traj_selector.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(
            self.traj_selector, 2, 0, QtCore.Qt.AlignmentFlag.AlignHCenter
        )

        self.traj_selector.sigValueChanged.connect(self.traj_to_render)

        # focus selector (each agent + map)
        focus_text = QtWidgets.QLabel("Focus on:")
        self.layout.addWidget(focus_text, 1, 1, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.focus_selector = pg.ComboBox(
            parent=self.canvas, items=env.agents + ["Map"], default="Map"
        )
        self.focus_selector.setFixedSize(100, 20)
        self.layout.addWidget(
            self.focus_selector, 2, 1, QtCore.Qt.AlignmentFlag.AlignHCenter
        )

        # focus selector (each agent + map)
        playback_text = QtWidgets.QLabel("Playback Speed:")
        self.layout.addWidget(playback_text, 1, 2, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.speed_selector = pg.ComboBox(
            parent=self.canvas, items=["0.5", "1.0", "2.0", "5.0"], default="1.0"
        )
        self.speed_selector.setFixedSize(100, 20)
        self.layout.addWidget(
            self.speed_selector, 2, 2, QtCore.Qt.AlignmentFlag.AlignHCenter
        )

        # buttons
        self.buttons = [
            QtWidgets.QPushButton("<< 1s"),
            QtWidgets.QPushButton("Play/Pause"),
            QtWidgets.QPushButton("1s >>"),
        ]
        for i, b in enumerate(self.buttons):
            b.setFixedSize(100, 20)
            self.layout.addWidget(b, 2, 3 + i, QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.buttons[0].clicked.connect(self.rewind)
        self.buttons[1].clicked.connect(self.play_pause)
        self.buttons[2].clicked.connect(self.fast_forward)

        # Disable interactivity
        self.canvas.setMouseEnabled(x=False, y=False)  # Disable mouse panning & zooming
        self.canvas.hideButtons()  # Disable corner auto-scale button
        self.canvas.setMenuEnabled(False)  # Disable right-click context menu

        legend = self.canvas.addLegend()  # This doesn't disable legend interaction
        # Override both methods responsible for mouse events
        legend.mouseDragEvent = lambda *args, **kwargs: None
        legend.hoverEvent = lambda *args, **kwargs: None

        # Remove axes
        self.canvas.hideAxis("bottom")
        self.canvas.hideAxis("left")

        # setting plot window background color
        self.window.setBackground("w")

        # fps and time renderer
        self.clock = FrameCounter()

        self.fps_renderer = pg.TextItem("FPS: 0.0", anchor=(0, 1), color=(255, 0, 0))
        self.time_renderer = pg.TextItem("0.0", anchor=(1, 1), color=(255, 0, 0))

        self.canvas.addItem(self.fps_renderer)
        self.canvas.addItem(self.time_renderer)
        self.fps_renderer.setPos(0, 0)
        self.time_renderer.setPos(1, 0)

        # playback control states
        self.playing = False
        self.speed = 1
        self.focus_on = (
            0  # 0: agent 0,  1: agent 1, 2: agent 2, 3: agent 3, ... , n: map
        )

        if self.render_mode in ["human", "human_fast"]:
            self.clock.sigFpsUpdate.connect(
                lambda fps: self.fps_renderer.setText(f"FPS: {fps:.1f}")
            )

        # generate a list of random color pallets in hex
        colors_rgb = [
            ImageColor.getcolor("#%06x" % np.random.randint(0, 0xFFFFFF), "RGB")
            for i in range(100)
        ]
        self.car_colors = [
            colors_rgb[i % len(colors_rgb)] for i in range(self.env.num_agents)
        ]

        # callbacks for custom visualization, called at every rendering step
        self.callbacks = []

        # rendering mode
        if self.render_mode == "human":
            self.speed = 1
            self.window.show()
        elif self.render_mode == "human_fast":
            self.speed = 3
            self.window.show()
        elif self.render_mode == "rgb_array":
            self.exporter = ImageExporter(self.canvas)

        self.traj_initialized = False

        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.timer_callback)
        self.t.start(int(1000 / self.render_fps / self.speed))

    def add_renderer_callback(
        self, callback_fn: Callable[[TrajRenderer], None]
    ) -> None:
        """
        Add a custom callback for visualization.
        All the callbacks are called at every rendering step, after having rendered the map and the cars.

        Parameters
        ----------
        callback_fn : Callable[[TrajRenderer], None]
            callback function to be called at every rendering step
        """
        self.callbacks.append(callback_fn)

    def play_pause(self):
        """Toggle play/pause"""
        self.playing = not self.playing
        self.t.stop() if self.playing else self.t.start()
        print("play pause")

    def rewind(self):
        # rewind playback 1 second
        self.current_step = int((self.current_step - self.render_fps) % self.num_steps)
        print(f"rewind to {self.current_step}")

    def fast_forward(self):
        # fast foward playback 1 second
        self.current_step = int((self.current_step + self.render_fps) % self.num_steps)
        print(f"fast forward to step {self.current_step}")

    def traj_to_render(self, spinbox):
        """Set the trajectory to render"""
        self.current_traj = int(spinbox.value())
        print(f"Trajectory to render: {self.current_traj}")

    def timer_callback(self):
        """Timer callback for rendering"""
        if not self.traj_initialized:
            return

        self.current_step = (self.current_step + 1) % self.num_steps
        current_speed = float(self.speed_selector.currentText())
        if self.speed != current_speed:
            self.t.setInterval(int((1000 / self.render_fps) / current_speed))
            self.speed = current_speed
        self.clock.update()
        self.time_renderer.setText(
            f"Sim time: {self.sim_time:.2f}, focus on {self.focus_selector.currentText()}"
        )
        self.sim_time += 1.0 / self.render_fps

        # update focus
        focus = self.focus_selector.currentText()
        if focus == "Map":
            self.canvas.getPlotItem().autoRange()
            self.focus_on = "Map"
        else:
            self.focus_on = self.env.agents.index(focus)
            ego_x = self.traj[self.current_step, self.current_traj, self.focus_on, 0]
            ego_y = self.traj[self.current_step, self.current_traj, self.focus_on, 1]
            try:
                self.canvas.getPlotItem().setXRange(ego_x - 10, ego_x + 10)
                self.canvas.getPlotItem().setYRange(ego_y - 10, ego_y + 10)
            except ValueError:
                pass

        # update car vertices
        for i in range(self.num_agents):
            vertices = get_vertices(
                self.traj[self.current_step, self.current_traj, i, [0, 1, 4]],
                self.env.params.length,
                self.env.params.width,
            )
            vertices = vertices[[0, 3, 2, 1], :]
            vertices = np.vstack([vertices, vertices[0]])
            self.chassis[i].setData(vertices[:, 0], vertices[:, 1])
        # update wheel vertices
        for i in range(self.num_agents):
            fl_vertices = _get_tire_vertices(
                self.traj[self.current_step, self.current_traj, i, [0, 1, 4]],
                self.env.params.length,
                self.env.params.width,
                self.wheel_width,
                self.wheel_length,
                True,
                self.traj[self.current_step, self.current_traj, i, 2],
            )
            self.wheels[i][0].setData(fl_vertices[:, 0], fl_vertices[:, 1])
            fr_vertices = _get_tire_vertices(
                self.traj[self.current_step, self.current_traj, i, [0, 1, 4]],
                self.env.params.length,
                self.env.params.width,
                self.wheel_width,
                self.wheel_length,
                False,
                self.traj[self.current_step, self.current_traj, i, 2],
            )
            self.wheels[i][1].setData(fr_vertices[:, 0], fr_vertices[:, 1])

    def render(self, trajectory: np.ndarray) -> Optional[np.ndarray]:
        """
        Initializes the rendering, starts the timer

        Returns
        -------
        Optional[np.ndarray]
            if render_mode is "rgb_array", returns the rendered frame as an array
        """
        # get sizes
        self.num_steps, self.num_envs, self.num_agents, self.num_states = (
            trajectory.shape
        )
        self.traj_initialized = True
        self.traj = trajectory
        self.traj_selector.setMaximum(self.num_envs)

        # initialize the render for the cars
        self.chassis = []
        self.wheels = []
        for i in range(self.num_agents):
            vertices = get_vertices(
                self.traj[self.current_step, self.current_traj, i, [0, 1, 4]],
                self.env.params.length,
                self.env.params.width,
            )
            vertices = vertices[[0, 3, 2, 1], :]
            vertices = np.vstack([vertices, vertices[0]])  # close the loop
            self.chassis.append(
                self.canvas.getPlotItem().plot(
                    vertices[:, 0],
                    vertices[:, 1],
                    pen=pg.mkPen(color=(0, 0, 0), width=self.thickness),
                    fillLevel=0,
                    brush=self.car_colors[i],
                ),
            )
            
            fl_vertices = _get_tire_vertices(
                self.traj[self.current_step, self.current_traj, i, [0, 1, 4]],
                self.env.params.length,
                self.env.params.width,
                self.wheel_width,
                self.wheel_length,
                True,
                self.traj[self.current_step, self.current_traj, i, 2],
            )
            fl_wheel = self.canvas.getPlotItem().plot(
                fl_vertices[:, 0],
                fl_vertices[:, 1],
                pen=pg.mkPen(color=(0, 0, 0), width=self.thickness),
                fillLevel=0,
                brush=(0, 0, 0),
            )
            fr_vertices = _get_tire_vertices(
                self.traj[self.current_step, self.current_traj, i, [0, 1, 4]],
                self.env.params.length,
                self.env.params.width,
                self.wheel_width,
                self.wheel_length,
                False,
                self.traj[self.current_step, self.current_traj, i, 2],
            )
            fr_wheel = self.canvas.getPlotItem().plot(
                fr_vertices[:, 0],
                fr_vertices[:, 1],
                pen=pg.mkPen(color=(0, 0, 0), width=self.thickness),
                fillLevel=0,
                brush=(0, 0, 0),
            )

            wheels = [fl_wheel, fr_wheel]
            self.wheels.append(wheels)

        # call render start
        if self.render_mode in ["human", "human_fast"]:
            pg.exec()

        else:
            # rgb_array mode => extract the frame from the canvas
            qImage = self.exporter.export(toBytes=True)

            width = qImage.width()
            height = qImage.height()

            ptr = qImage.bits()
            ptr.setsize(height * width * 4)
            frame = np.array(ptr).reshape(height, width, 4)  #  Copies the data

            # advance frame counter
            if self.current_step >= self.num_steps:
                return None
            else:
                self.current_step += 1
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

    def render_vertices(
        self, vertices: np.ndarray, color: Optional[tuple[int, int, int]] = (0, 0, 255)
    ) -> pg.PlotDataItem:
        """
        Render a sequence of vertices on screen.

        Parameters
        ----------
        vertices : np.ndarray
            list of vertices to render
        color : Optional[tuple[int, int, int]], optional
            color as rgb tuple, by default blue (0, 0, 255)
        """
        return self.canvas.plot(
            vertices[:, 0],
            vertices[:, 1],
            pen=None,
            symbol="o",
            symbolPen=pg.mkPen(color=color, width=0),
            symbolBrush=pg.mkBrush(color=color, width=0),
            symbolSize=5,
        )

    def close(self) -> None:
        """
        Close the rendering environment.
        """
        self.app.exit()
        self.app.closeAllWindows()
