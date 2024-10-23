# world/components/blocks/interactive/goal_block.py

import pygame
from world.components.blocks.block import Block

class GoalBlock(Block):
    def __init__(self, x, y, width=40, height=40, color=(0, 255, 0)):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.color = color  # Green color for the goal
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        # GoalBlock doesn't need to update
        pass

    def interact(self, agent):
        # Define what happens when the agent reaches the goal
        pass
