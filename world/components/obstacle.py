import pygame
from agent.agent import Agent

BLOCK_COLOR = (255, 0, 0)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


# Block class
class Block(pygame.sprite.Sprite):
    def __init__(self, x, y, width=40, height=40):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(BLOCK_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
