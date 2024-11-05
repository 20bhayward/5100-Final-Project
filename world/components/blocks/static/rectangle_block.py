# world/blocks/rectangle_block.py

import pygame
from world.components.blocks.static.static_block import StaticBlock

class RectangleBlock(StaticBlock):
    def __init__(self, x, y, width=80, height=40,  color=(255, 255, 255)):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.color = color
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
