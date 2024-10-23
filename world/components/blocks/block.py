# world/blocks/block.py

import pygame
from abc import ABC, abstractmethod

class Block(pygame.sprite.Sprite, ABC):
    def __init__(self, x, y):
        super().__init__()
        self.image = None
        self.rect = None
        self.mask = None
        self.rect = pygame.Rect(x, y, 0, 0)  # Placeholder; subclasses will set size

    @abstractmethod
    def update(self):
        pass
