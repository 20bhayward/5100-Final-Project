# Create a new water block component in world/components/blocks/static/water_block.py
import pygame
from world.components.blocks.block import Block

class WaterBlock(Block):
    """
    A class representing a WaterBlock in the game.

    Attributes:
        image (pygame.Surface): The image representing the water block.
        rect (pygame.Rect): The rectangle representing the position and dimensions of the water block.
        mask (pygame.mask.Mask): The mask used for collision detection.

    Methods:
        __init__(x, y, width=100, height=120, color=(28,163,236)):
            Initializes the WaterBlock with the given position, dimensions, and color.
        update():
            Updates the state of the WaterBlock. Currently a placeholder.
        interact(agent):
            Defines the interaction between the WaterBlock and an agent. Currently a placeholder.
    """
    
    def __init__(self, x, y, width=100, height=120, color=(28,163,236)):
        """
        Initializes a WaterBlock instance.

        Args:
            x (int): The x-coordinate of the block.
            y (int): The y-coordinate of the block.
            width (int, optional): The width of the block. Defaults to 100.
            height (int, optional): The height of the block. Defaults to 120.
            color (tuple, optional): The color of the block in RGB format. Defaults to (28, 163, 236).
        """
        super().__init__(x, y)
        self.image = pygame.image.load("world/assets/New Piskel-2.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        """
        Update the state of the water block. This method is intended to be overridden
        by subclasses to provide specific update functionality.
        """
        pass

    def interact(self, agent):
        """
        Handles the interaction between the agent and the water block.

        Parameters:
        agent (object): The agent that interacts with the water block.

        Returns:
        None
        """
        # Define what happens when the agent hits the trap
        pass