# world/blocks/block.py

import pygame
from abc import ABC, abstractmethod

class Block(pygame.sprite.Sprite, ABC):
    """
    A base class for all block types in the game, inheriting from pygame's Sprite class and the ABC class for abstract methods.

    Attributes:
        image (pygame.Surface): The image representing the block.
        rect (pygame.Rect): The rectangular area of the block.
        mask (pygame.mask.Mask): The mask for pixel-perfect collision detection.

    Args:
        x (int): The x-coordinate of the block's position.
        y (int): The y-coordinate of the block's position.

    Methods:
        update(): An abstract method that must be implemented by subclasses to update the block's state.
    """
    def __init__(self, x, y):
        """
        Initializes a Block instance.

        Args:
            x (int): The x-coordinate of the block.
            y (int): The y-coordinate of the block.
        """
        super().__init__()
        self.image = None
        self.rect = None
        self.mask = None
        self.rect = pygame.Rect(x, y, 0, 0)  # Placeholder; subclasses will set size

    @abstractmethod
    def update(self):
        """
        Update the state of the block. This method should be overridden by subclasses
        to provide specific update functionality.
        """
        pass
