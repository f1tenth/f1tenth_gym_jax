from __future__ import annotations

import jax
import pyqtgraph as pg

from ..collision_models import get_trmtx


class TextObject:
    """
    Class to display text on the screen at a given position.

    Attributes
    ----------
    font : pygame.font.Font
        font object
    position : str | tuple
        position of the text on the screen
    text : pygame.Surface
        text surface to be displayed
    """

    def __init__(
        self,
        position: str | tuple,
        relative_font_size: int = 16,
        font_name: str = "Arial",
        parent: pg.PlotWidget = None,
    ) -> None:
        """
        Initialize text object.

        Parameters
        ----------
        position : str | tuple
            position of the text on the screen
        relative_font_size : int, optional
            font size relative to the window shape, by default 32
        font_name : str, optional
            font name, by default "Arial"
        """
        self.position = position

        self.text_label = pg.LabelItem(
            "",
            parent=parent,
            size=str(relative_font_size) + "pt",
            family=font_name,
            color=(125, 125, 125),
        )  # create text label
        # Get the position and offset of the text
        position_tuple = self._position_resolver(self.position)
        offset_tuple = self._offset_resolver(self.position, self.text_label)
        # Set the position and offset of the text
        self.text_label.anchor(
            itemPos=position_tuple, parentPos=position_tuple, offset=offset_tuple
        )

    def _position_resolver(self, position: str | tuple[int, int]) -> tuple[int, int]:
        """
        This function takes strings like "bottom center" and converts them into a location for the text to be displayed.
        If position is tuple, then passthrough.

        Parameters
        ----------
        position : str | tuple
            position of the text on the screen

        Returns
        -------
        tuple
            position of the text on the screen

        Raises
        ------
        ValueError
            if position is not a tuple or a string
        NotImplementedError
            if position is a string but not implemented
        """
        if isinstance(position, tuple) and len(position) == 2:
            return int(position[0]), int(position[1])
        elif isinstance(position, str):
            position = position.lower()
            if position == "bottom_right":
                return (1, 1)
            elif position == "bottom_left":
                return (0, 1)
            elif position == "bottom_center":
                return (0.5, 1)
            elif position == "top_right":
                return (1, 0)
            elif position == "top_left":
                return (0, 0)
            elif position == "top_center":
                return (0.5, 0)
            else:
                raise NotImplementedError(f"Position {position} not implemented.")
        else:
            raise ValueError(
                f"Position expected to be a tuple[int, int] or a string. Got {position}."
            )

    def _offset_resolver(
        self, position: str | tuple[int, int], text_label: pg.LabelItem
    ) -> tuple[int, int]:
        """
        This function takes strings like "bottom center" and converts them into a location for the text to be displayed.
        If position is tuple, then passthrough.

        Parameters
        ----------
        position : str | tuple
            position of the text on the screen

        Returns
        -------
        tuple
            position of the text on the screen

        Raises
        ------
        ValueError
            if position is not a tuple or a string
        NotImplementedError
            if position is a string but not implemented
        """
        if isinstance(position, tuple) and len(position) == 2:
            return int(position[0]), int(position[1])
        elif isinstance(position, str):
            position = position.lower()
            if position == "bottom_right":
                return (-text_label.width(), 0)
            elif position == "bottom_left":
                return (0, 0)
            elif position == "bottom_center":
                return (-text_label.width() / 2, 0)
            elif position == "top_right":
                return (-text_label.width(), 0)
            elif position == "top_left":
                return (0, 0)
            elif position == "top_center":
                return (-text_label.width() / 2, 0)
            else:
                raise NotImplementedError(f"Position {position} not implemented.")
        else:
            raise ValueError(
                f"Position expected to be a tuple[int, int] or a string. Got {position}."
            )

    def render(self, text: str) -> None:
        """
        Render text on the screen.

        Parameters
        ----------
        text : str
            text to be displayed
        """
        self.text_label.setText(text)


@jax.jit
def _get_tire_vertices(pose_arr, length, width, tire_width, tire_length, fl, steering):
    """
    Utility function to return vertices of the car's tire given pose and size

    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width

    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    working_width = jax.lax.select(fl, width, -width)
    working_tire_width = jax.lax.select(fl, tire_width, -tire_width)

    H_shift = get_trmtx(
        jax.numpy.array(
            [
                -(length / 2 - tire_length / 2),
                -(working_width / 2 - working_tire_width / 2),
                0,
            ]
        )
    )
    H_steer = get_trmtx(jax.numpy.array([0, 0, steering]))
    H_back = get_trmtx(
        jax.numpy.array(
            [
                length / 2 - tire_length / 2,
                working_width / 2 - working_tire_width / 2,
                0,
            ]
        )
    )
    H = get_trmtx(pose_arr)
    H = H.dot(H_back).dot(H_steer).dot(H_shift)
    fl = H.dot(
        jax.numpy.asarray([[length / 2], [working_width / 2], [0.0], [1.0]])
    ).flatten()
    fr = H.dot(
        jax.numpy.asarray(
            [[length / 2], [working_width / 2 - working_tire_width], [0.0], [1.0]]
        )
    ).flatten()
    rr = H.dot(
        jax.numpy.asarray(
            [
                [length / 2 - tire_length],
                [working_width / 2 - working_tire_width],
                [0.0],
                [1.0],
            ]
        )
    ).flatten()
    rl = H.dot(
        jax.numpy.asarray(
            [[length / 2 - tire_length], [working_width / 2], [0.0], [1.0]]
        )
    ).flatten()
    rl = rl / rl[3]
    rr = rr / rr[3]
    fl = fl / fl[3]
    fr = fr / fr[3]
    vertices = jax.numpy.asarray(
        [
            [rl[0], rl[1]],
            [fl[0], fl[1]],
            [fr[0], fr[1]],
            [rr[0], rr[1]],
            [rl[0], rl[1]],
        ]
    )

    return vertices
