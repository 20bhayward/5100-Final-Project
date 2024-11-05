# world/blocks/triangle_block.py

import pygame
from world.components.blocks.static.static_block import StaticBlock

class TriangleBlock(StaticBlock):
    def __init__(self, x, y, base=60, height=60,  color=(255, 255, 255)):
        super().__init__(x, y)
        self.base = base
        self.height = height
        self.color = color
        self.image = pygame.Surface([base, height], pygame.SRCALPHA)
        pygame.draw.polygon(
            self.image,
            color,
            [(0, self.height), (self.base, self.height), (0, 0)]
        )
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
