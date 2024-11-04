import jax
import jax.numpy as jnp
import numpy as np
import pyglet
from pyglet import shapes
from pyglet.gl import *

from ..track import Track
from ..collision_models import get_vertices

class TrajectoryPlayer:
    def __init__(self, track: Track, window_width: int=800, window_height: int=600):
        """
        Initialize the player with the map and trajectories.

        Args:
            track (Track): track used.
            window_width (int): Width of the window.
            window_height (int): Height of the window.
        """
        # Initialize window
        self.window = pyglet.window.Window(width=window_width, height=window_height)
        
        # Load map image as background
        self.background_sprite = pyglet.sprite.Sprite(track.occ_map)
        
        # Store trajectories and initialize rectangle shapes in a batch
        self.batch = pyglet.graphics.Batch()
        
        
        # Playback control
        self.time_step = 0
        self.playing = False
        self.speed = 1  # Controls speed of playback (1 = normal, 2 = double, etc.)
        
        # Button positions and state
        self.buttons = {
            'play_pause': shapes.Rectangle(10, 10, 80, 30, color=(0, 200, 0), batch=self.batch),
            'faster': shapes.Rectangle(100, 10, 80, 30, color=(0, 100, 200), batch=self.batch),
            'slower': shapes.Rectangle(190, 10, 80, 30, color=(200, 100, 0), batch=self.batch),
        }
        self.button_texts = {
            'play_pause': pyglet.text.Label('Play', x=50, y=25, anchor_x='center', batch=self.batch),
            'faster': pyglet.text.Label('Faster', x=140, y=25, anchor_x='center', batch=self.batch),
            'slower': pyglet.text.Label('Slower', x=230, y=25, anchor_x='center', batch=self.batch),
        }

    def update(self, dt):
        """Update positions of rectangles according to trajectories."""
        if not self.playing:
            return  # Pause if not playing

        # Update each rectangle's position in its trajectory
        for idx, (rectangle, trajectory) in enumerate(zip(self.rectangles, self.trajectories)):
            if self.time_step < len(trajectory):
                new_x, new_y = trajectory[self.time_step]
                rectangle.x = new_x
                rectangle.y = new_y

        # Advance the trajectory index, respecting playback speed
        self.time_step += self.speed

    def draw(self):
        """Draw the background map, rectangles, and buttons."""
        self.window.clear()
        self.background_sprite.draw()
        self.batch.draw()

    def toggle_play_pause(self):
        """Toggle between play and pause states."""
        self.playing = not self.playing
        self.button_texts['play_pause'].text = 'Pause' if self.playing else 'Play'

    def increase_speed(self):
        """Increase playback speed."""
        self.speed += 1

    def decrease_speed(self):
        """Decrease playback speed, to a minimum of 1."""
        self.speed = max(1, self.speed - 1)

    def check_button_click(self, x, y):
        """Check if a button was clicked."""
        if self.buttons['play_pause'].x < x < self.buttons['play_pause'].x + self.buttons['play_pause'].width and \
           self.buttons['play_pause'].y < y < self.buttons['play_pause'].y + self.buttons['play_pause'].height:
            self.toggle_play_pause()
        elif self.buttons['faster'].x < x < self.buttons['faster'].x + self.buttons['faster'].width and \
             self.buttons['faster'].y < y < self.buttons['faster'].y + self.buttons['faster'].height:
            self.increase_speed()
        elif self.buttons['slower'].x < x < self.buttons['slower'].x + self.buttons['slower'].width and \
             self.buttons['slower'].y < y < self.buttons['slower'].y + self.buttons['slower'].height:
            self.decrease_speed()

    def render_trajectory(self, trajectory):
        """Render a trajectory as a series of rectangles."""
        rectangles = []
        for x, y in trajectory:
            rectangle = shapes.Rectangle(x, y, 10, 10, color=(255, 0, 0), batch=self.batch)
            rectangles.append(rectangle)
        return rectangles

    def run(self):
        """Run the player."""
        # Schedule the update function
        pyglet.clock.schedule_interval(self.update, 1/30)  # 30 FPS update rate
        
        # Set up window draw and mouse events
        @self.window.event
        def on_draw():
            self.draw()

        @self.window.event
        def on_mouse_press(x, y, button, modifiers):
            self.check_button_click(x, y)

        # Start the pyglet application
        pyglet.app.run()

# Example usage
if __name__ == '__main__':
    # Define some sample trajectories for two rectangles
    sample_trajectories = [
        [(100 + i * 5, 100 + i * 5) for i in range(100)],  # Diagonal movement
        [(300 - i * 5, 200 + i * 3) for i in range(100)]   # Diagonal in another direction
    ]
    
    # Path to map image (replace 'map.png' with your actual map file)
    map_image_path = 'map.png'
    
    # Create the player and run it
    player = TrajectoryPlayer(map_image_path, sample_trajectories)
    player.run()
