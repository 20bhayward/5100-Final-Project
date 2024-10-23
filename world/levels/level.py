# world/levels/level.py

import pygame
from abc import ABC, abstractmethod

class Level(ABC):
    def __init__(self):
        self.block_list = pygame.sprite.Group()
        self.goal_list = pygame.sprite.Group()
        self.trap_list = pygame.sprite.Group()
        self.enemy_list = pygame.sprite.Group()
        self.all_sprites_list = pygame.sprite.Group()
        self.width = 2000  # Default level width
        self.height = 600  # Default level height
        self.create_level()

    @abstractmethod
    def create_level(self):
        pass

    def get_blocks(self):
        return self.block_list

    def get_all_sprites(self):
        return self.all_sprites_list
