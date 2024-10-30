# Create a new water block component in world/components/blocks/static/water_block.py
import pygame
from world.components.blocks.block import Block

class WaterBlock(Block):
    def __init__(self, x, y, width=40, height=40, color=(28,163,236)):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.color = color  # Red color for the trap
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        # WaterBlock might have animation or other behaviors
        pass

    def interact(self, agent):
        # Define what happens when the agent hits the trap
        pass