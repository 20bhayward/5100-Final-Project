# world/components/blocks/interactive/moving_block.py

import pygame
from world.components.blocks.interactive.interactive_block import InteractiveBlock

class MovingBlock(InteractiveBlock):
    """
    A class representing a moving block in the game world.

    Attributes:
        speed (int): The speed at which the block moves.
        direction (str): The direction of movement, either 'horizontal' or 'vertical'.
        image (pygame.Surface): The image of the block.
        rect (pygame.Rect): The rectangle representing the block's position and size.
        mask (pygame.mask.Mask): The mask for collision detection.
        start_pos (int): The starting position of the block's movement.
        end_pos (int): The ending position of the block's movement.

    Methods:
        update():
            Updates the block's position based on its speed and direction.
            Reverses the direction when the block reaches its movement bounds.
        
        interact(agent):
            Defines interaction with the agent if necessary.
    """

    def __init__(self, x, y, width, height, color, speed, direction, start_pos=None, end_pos=None):
        """
        Initializes a MovingBlock object.

        Args:
            x (int): The x-coordinate of the block.
            y (int): The y-coordinate of the block.
            width (int): The width of the block.
            height (int): The height of the block.
            color (tuple): The color of the block (not used in the current implementation).
            speed (int): The speed at which the block moves.
            direction (str): The direction of movement, either 'horizontal' or 'vertical'.
            start_pos (int, optional): The starting position of the block's movement. Defaults to None.
            end_pos (int, optional): The ending position of the block's movement. Defaults to None.
        """
        super().__init__(x, y)
        self.speed = speed
        self.direction = direction  # 'horizontal' or 'vertical'
        self.image = pygame.image.load("world/assets/ground.png")
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

        # Set movement bounds
        if self.direction == 'horizontal':
            self.start_pos = start_pos if start_pos is not None else self.rect.left
            self.end_pos = end_pos if end_pos is not None else self.rect.right
        elif self.direction == 'vertical':
            self.start_pos = start_pos if start_pos is not None else self.rect.top
            self.end_pos = end_pos if end_pos is not None else self.rect.bottom

    def update(self):
        """
        Update the position of the moving block based on its direction and speed.
        
        If the block is moving horizontally, it will update its x-coordinate by adding the speed.
        If the block reaches the custom bounds (start_pos or end_pos), it will reverse its direction.
        
        If the block is moving vertically, it will update its y-coordinate by adding the speed.
        If the block reaches the custom bounds (start_pos or end_pos), it will reverse its direction.
        
        Attributes:
            direction (str): The direction of the block's movement ('horizontal' or 'vertical').
            rect (pygame.Rect): The rectangle representing the block's position and size.
            speed (int): The speed at which the block moves.
            start_pos (int): The starting position of the block.
            end_pos (int): The ending position of the block.
        """
        if self.direction == 'horizontal':
            self.rect.x += self.speed
            # Reverse direction at custom bounds
            if self.rect.left <= self.start_pos or self.rect.right >= self.end_pos:
                self.speed = -self.speed
        elif self.direction == 'vertical':
            self.rect.y += self.speed
            # Reverse direction at custom bounds
            if self.rect.top <= self.start_pos or self.rect.bottom >= self.end_pos:
                self.speed = -self.speed

    def interact(self, agent):
        """
        Handles the interaction between the moving block and an agent.

        Parameters:
        agent (object): The agent that interacts with the moving block.
        """
        # Define interaction with the agent if necessary
        pass
