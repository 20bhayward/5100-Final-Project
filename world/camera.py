# world/camera.py

import pygame
from gym_env import SCREEN_WIDTH, SCREEN_HEIGHT

class Camera:
    """
    A class to represent a camera that follows a target within a level.

    Attributes:
    -----------
    camera_rect : pygame.Rect
        A rectangle representing the camera's position and size.
    level_width : int
        The width of the level.
    level_height : int
        The height of the level.

    Methods:
    --------
    __init__(level_width, level_height):
        Initializes the camera with the level dimensions.
    apply(rect):
        Applies the camera's offset to a given rectangle.
    update(target):
        Updates the camera's position based on the target's position, 
        ensuring the camera stays within the level boundaries.
    """
    def __init__(self, level_width, level_height):
        """
        Initializes the Camera object with the given level dimensions.

        Args:
            level_width (int): The width of the level.
            level_height (int): The height of the level.
        """
        self.camera_rect = pygame.Rect(0, 0, level_width, level_height)
        self.level_width = level_width
        self.level_height = level_height

    def apply(self, rect):
        """
        Adjusts the given rectangle's position based on the camera's current position.

        Args:
            rect (pygame.Rect): The rectangle to be adjusted.

        Returns:
            pygame.Rect: A new rectangle with its position adjusted according to the camera's position.
        """
        return rect.move(self.camera_rect.topleft)

    def update(self, target):
        """
        Update the camera position based on the target's position.

        This method adjusts the camera's position to center on the target,
        while ensuring that the camera does not scroll beyond the boundaries
        of the level.

        Args:
            target (pygame.sprite.Sprite): The target sprite that the camera
                           should follow. The target is expected
                           to have a 'rect' attribute.

        Attributes:
            self.camera_rect (pygame.Rect): The updated rectangle representing
                            the camera's position and size.
        """
        x = -target.rect.x + int(SCREEN_WIDTH / 2)
        y = -target.rect.y + int(SCREEN_HEIGHT / 2)

        # Limit scrolling to level boundaries
        x = min(0, x)  # Left
        x = max(-(self.level_width - SCREEN_WIDTH), x)  # Right
        y = min(0, y)  # Top
        y = max(-(self.level_height - SCREEN_HEIGHT), y)  # Bottom

        self.camera_rect = pygame.Rect(x, y, self.level_width, self.level_height)
