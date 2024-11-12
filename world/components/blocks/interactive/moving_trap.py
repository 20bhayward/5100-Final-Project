# world/components/blocks/interactive/moving_block.py

import pygame
from world.components.blocks.interactive.interactive_block import InteractiveBlock

class MovingTrap(InteractiveBlock):
    """
    A class representing a moving trap in the game.

    Attributes:
    -----------
    speed : int
        The speed at which the trap moves.
    direction : str
        The direction of movement, either 'horizontal' or 'vertical'.
    image : pygame.Surface
        The image of the trap.
    rect : pygame.Rect
        The rectangle representing the trap's position and size.
    mask : pygame.mask.Mask
        The mask for pixel-perfect collision detection.
    start_pos : int
        The starting position of the trap's movement.
    end_pos : int
        The ending position of the trap's movement.

    Methods:
    --------
    update():
        Updates the trap's position and reverses its direction at the movement bounds.
    interact(agent):
        Defines interaction with the agent if necessary.
    """
    
    def __init__(self, x, y, width, height, color, speed, direction, start_pos=None, end_pos=None):
        """
        Initialize a MovingTrap object.

        Args:
            x (int): The x-coordinate of the trap's initial position.
            y (int): The y-coordinate of the trap's initial position.
            width (int): The width of the trap.
            height (int): The height of the trap.
            color (tuple): The color of the trap (not used in the current implementation).
            speed (int): The speed at which the trap moves.
            direction (str): The direction of the trap's movement ('horizontal' or 'vertical').
            start_pos (int, optional): The starting position of the trap's movement. Defaults to None.
            end_pos (int, optional): The ending position of the trap's movement. Defaults to None.
        """
        super().__init__(x, y)
        self.speed = speed
        self.direction = direction  # 'horizontal' or 'vertical'
        self.image = pygame.image.load("world/assets/spike-sprite.png")
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
        Update the position of the moving trap based on its direction and speed.
        
        The trap moves either horizontally or vertically and reverses direction
        when it reaches the custom bounds defined by start_pos and end_pos.
        
        Attributes:
            direction (str): The direction of movement ('horizontal' or 'vertical').
            rect (pygame.Rect): The rectangle representing the trap's position and size.
            speed (int): The speed at which the trap moves.
            start_pos (int): The starting position boundary for reversing direction.
            end_pos (int): The ending position boundary for reversing direction.
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
        Interact with the agent.

        This method defines the interaction between the moving trap and the agent.
        It can be overridden to specify the behavior when the agent interacts with the trap.

        Parameters:
        agent (object): The agent that interacts with the moving trap.
        """
        # Define interaction with the agent if necessary
        pass
