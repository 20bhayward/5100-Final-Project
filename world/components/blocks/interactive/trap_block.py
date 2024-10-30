# world/components/blocks/trap_block.py

import pygame
from world.components.blocks.block import Block

class TrapBlock(Block):
    def __init__(self, x, y, width=50, height=30, color=(255, 0, 0)):
        super().__init__(x, y)
        self.image = pygame.image.load("world/assets/spike-sprite.png")
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        # TrapBlock might have animation or other behaviors
        pass

    def interact(self, agent):
        # Define what happens when the agent hits the trap
        pass
