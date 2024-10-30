# Create a new water block component in world/components/blocks/static/water_block.py
import pygame
from world.components.blocks.block import Block

class WaterBlock(Block):
    def __init__(self, x, y, width=100, height=120, color=(28,163,236)):
        super().__init__(x, y)
        self.image = pygame.image.load("world/assets/New Piskel-2.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        pass

    def interact(self, agent):
        # Define what happens when the agent hits the trap
        pass