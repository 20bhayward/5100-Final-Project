# world/components/blocks/interactive/moving_block.py

import pygame
from world.components.blocks.interactive.interactive_block import InteractiveBlock

class MovingBlock(InteractiveBlock):
    """
    A class representing a moving block in the game world.
    """

    def __init__(self, x, y, width, height, color, speed, direction, start_pos=None, end_pos=None):
        """
        Initializes a MovingBlock object.
        """
        super().__init__(x, y)
        self.speed = speed
        self.direction = direction  # 'horizontal' or 'vertical'
        self.image = pygame.image.load("world/assets/ground.png")
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
        self.is_moving = True

        # Set movement bounds and initial positions
        if self.direction == 'horizontal':
            self.start_pos = start_pos if start_pos is not None else self.rect.left
            self.end_pos = end_pos if end_pos is not None else self.rect.right
            self.start_y = y
        elif self.direction == 'vertical':
            self.start_pos = start_pos if start_pos is not None else self.rect.top
            self.end_pos = end_pos if end_pos is not None else self.rect.bottom
            self.start_y = y

    def update(self):
        """
        Update the position of the moving block based on its direction and speed.
        """
        if self.direction == 'horizontal':
            self.rect.x += self.speed
            # Reverse direction at custom bounds
            if self.rect.left <= self.start_pos or self.rect.right >= self.end_pos:
                self.speed = -self.speed
        elif self.direction == 'vertical':
            self.rect.y += self.speed
            # Reverse direction at custom bounds
            if self.rect.top <= self.start_pos or self.rect.bottom >= self.end_pos:
                self.speed = -self.speed

    def interact(self, agent):
        """
        Handles the interaction between the moving block and an agent.
        """
        # Define interaction with the agent if necessary
        pass
