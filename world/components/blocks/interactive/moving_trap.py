# world/components/blocks/interactive/moving_block.py

import pygame
from world.components.blocks.interactive.interactive_block import InteractiveBlock

class MovingTrap(InteractiveBlock):
    def __init__(self, x, y, width, height, color, speed, direction, start_pos=None, end_pos=None):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.color = color
        self.speed = speed
        self.direction = direction  # 'horizontal' or 'vertical'
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

        # Set movement bounds
        if self.direction == 'horizontal':
            self.start_pos = start_pos if start_pos is not None else self.rect.left
            self.end_pos = end_pos if end_pos is not None else self.rect.right
        elif self.direction == 'vertical':
            self.start_pos = start_pos if start_pos is not None else self.rect.top
            self.end_pos = end_pos if end_pos is not None else self.rect.bottom

    def update(self):
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
        # Define interaction with the agent if necessary
        pass
