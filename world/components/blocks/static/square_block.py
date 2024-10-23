# world/blocks/square_block.py

import pygame
from world.components.blocks.static.static_block import StaticBlock

class SquareBlock(StaticBlock):
    def __init__(self, x, y, size=40,  color=(255, 255, 255)):
        super().__init__(x, y)
        self.size = size
        self.color = color
        self.image = pygame.Surface([size, size])
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
