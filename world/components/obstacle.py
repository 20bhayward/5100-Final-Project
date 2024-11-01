# world/obstacle.py

import pygame
import random

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
OBSTACLE_WIDTH = 30
# Set a fixed value for the height of the obstacles
OBSTACLE_MIN_HEIGHT = 40
OBSTACLE_MAX_HEIGHT = 110
GROUND_LEVEL = SCREEN_HEIGHT  # Using AGENT_SIZE value from agent.py
RED = (255, 0, 0)

# class Obstacle:
#     def __init__(self):
#         # Generate a random height for the obstacle
#         height = random.randint(OBSTACLE_MIN_HEIGHT, OBSTACLE_MAX_HEIGHT)
#         # Position the obstacle at the ground level minus its height
#         self.rect = pygame.Rect(SCREEN_WIDTH, GROUND_LEVEL - height, OBSTACLE_WIDTH, height)

#     def update(self):
#         self.rect.x -= 5  # Move the obstacle left at a speed of 5 pixels per update

#     def draw(self, screen):
#         pygame.draw.rect(screen, RED, self.rect)

class Obstacle(pygame.sprite.Sprite):
    """
    A class to represent an obstacle in the game world.

    Attributes
    ----------
    x : int
        The x-coordinate of the obstacle.
    y : int
        The y-coordinate of the obstacle.
    image : pygame.Surface
        The image representing the obstacle.
    rect : pygame.Rect
        The rectangular area of the obstacle.
    mask : pygame.mask.Mask
        The mask for pixel-perfect collision detection.

    Methods
    -------
    update():
        Placeholder method for future updates.
    """
    def __init__(self, x, y):
        super().__init__()
        self.image = None
        self.rect = None
        self.mask = None
        self.x = x
        self.y = y

    def update(self):
        # Placeholder for future updates
        pass
