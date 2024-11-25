# world/components/blocks/trap_block.py

import pygame
from world.components.blocks.block import Block

class TrapBlock(Block):
    """
    TrapBlock is a type of Block that represents a trap in the game world. It is an interactive block that can cause
    harm or trigger events when an agent interacts with it.

    Attributes:
        image (pygame.Surface): The image representing the trap block.
        rect (pygame.Rect): The rectangular area of the trap block.
        mask (pygame.mask.Mask): The mask used for pixel-perfect collision detection.

    Methods:
        __init__(x, y, width=50, height=30, color=(255, 0, 0)):
            Initializes the TrapBlock with a position, size, and color.
        update():
            Updates the state of the TrapBlock. This method can be used to handle animations or other behaviors.
        interact(agent):
            Defines the interaction behavior when an agent hits the trap.
    """
    
    def __init__(self, x, y, width=30, height=20, color=(255, 0, 0)):
        """
        Initializes a TrapBlock object.

        Args:
            x (int): The x-coordinate of the top-left corner of the block.
            y (int): The y-coordinate of the top-left corner of the block.
            width (int, optional): The width of the block. Defaults to 50.
            height (int, optional): The height of the block. Defaults to 30.
            color (tuple, optional): The color of the block in RGB format. Defaults to (255, 0, 0).
        """
        super().__init__(x, y)
        self.image = pygame.image.load("world/assets/spike-sprite.png")
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        """
        Update the state of the TrapBlock.

        This method is intended to handle any animations or behaviors
        that the TrapBlock might have. Currently, it does not perform
        any actions.
        """
        # TrapBlock might have animation or other behaviors
        pass

    def interact(self, agent):
        """
        Handles the interaction between the agent and the trap block.

        This method defines the behavior that occurs when the agent interacts with the trap block.

        Parameters:
        agent (object): The agent that interacts with the trap block.
        """
        # Define what happens when the agent hits the trap
        pass
