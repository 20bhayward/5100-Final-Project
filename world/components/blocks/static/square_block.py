# world/blocks/square_block.py

import pygame
from world.components.blocks.static.static_block import StaticBlock

class SquareBlock(StaticBlock):
    def __init__(self, x, y, size=40,  color=(175, 100, 32)):
        super().__init__(x, y)
        self.size = size
        self.color = color
        self.image = pygame.image.load("world/assets/ground.png").convert()
        self.image = pygame.transform.scale(self.image, (size, size))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
