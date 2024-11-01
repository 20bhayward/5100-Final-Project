# world/components/blocks/interactive/goal_block.py

import pygame
from world.components.blocks.block import Block

class GoalBlock(Block):
    """
    GoalBlock is a type of Block that represents a goal in the game. When an agent reaches this block, 
    a specific interaction can be triggered.

    Attributes:
        x (int): The x-coordinate of the block.
        y (int): The y-coordinate of the block.
        width (int): The width of the block. Default is 40.
        height (int): The height of the block. Default is 40.
        color (tuple): The color of the block in RGB format. Default is green (0, 255, 0).
        image (pygame.Surface): The image representing the goal block.
        rect (pygame.Rect): The rectangle area of the block.
        mask (pygame.mask.Mask): The mask for pixel-perfect collision detection.

    Methods:
        update():
            Updates the state of the GoalBlock. Currently, it does nothing as the GoalBlock doesn't need to update.
        interact(agent):
            Defines the interaction that occurs when an agent reaches the goal block. This method should be implemented 
            to specify what happens when the agent interacts with the goal.
    """
    def __init__(self, x, y, width=40, height=40, color=(0, 255, 0)):
        """
        Initializes a GoalBlock object.

        Args:
            x (int): The x-coordinate of the top-left corner of the block.
            y (int): The y-coordinate of the top-left corner of the block.
            width (int, optional): The width of the block. Defaults to 40.
            height (int, optional): The height of the block. Defaults to 40.
            color (tuple, optional): The color of the block in RGB format. Defaults to (0, 255, 0).

        Attributes:
            image (pygame.Surface): The image of the block, loaded and scaled.
            rect (pygame.Rect): The rectangle representing the block's position and size.
            mask (pygame.mask.Mask): The mask for pixel-perfect collision detection.
        """
        super().__init__(x, y)
        self.image = pygame.image.load('world/assets/new_door.jpeg').convert_alpha()
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        """
        Update method for GoalBlock.

        This method is intentionally left blank as GoalBlock does not require any updates.
        """
        # GoalBlock doesn't need to update
        pass

    def interact(self, agent):
        """
        Handles the interaction when the agent reaches the goal block.

        Parameters:
        agent (object): The agent that interacts with the goal block.
        """
        # Define what happens when the agent reaches the goal
        pass
