# world/components/blocks/interactive/goal_block.py

import pygame
from world.components.blocks.block import Block

class GoalBlock(Block):
    def __init__(self, x, y, width=40, height=40, color=(0, 255, 0)):
        super().__init__(x, y)
        self.image = pygame.image.load('world/assets/new_door.jpeg').convert_alpha()
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        # GoalBlock doesn't need to update
        pass

    def interact(self, agent):
        # Define what happens when the agent reaches the goal
        pass
