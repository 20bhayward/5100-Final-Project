# world/camera.py

import pygame

class Camera:
    def __init__(self, level_width, level_height):
        self.camera_rect = pygame.Rect(0, 0, level_width, level_height)
        self.level_width = level_width
        self.level_height = level_height

    def apply(self, rect):
        return rect.move(self.camera_rect.topleft)

    def update(self, target):
        x = -target.rect.x + int(SCREEN_WIDTH / 2)
        y = -target.rect.y + int(SCREEN_HEIGHT / 2)

        # Limit scrolling to level boundaries
        x = min(0, x)  # Left
        x = max(-(self.level_width - SCREEN_WIDTH), x)  # Right
        y = min(0, y)  # Top
        y = max(-(self.level_height - SCREEN_HEIGHT), y)  # Bottom

        self.camera_rect = pygame.Rect(x, y, self.level_width, self.level_height)
