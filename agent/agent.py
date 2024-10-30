# agent.py
import numpy as np
import pygame
from world.components.blocks.interactive.goal_block import GoalBlock

AGENT_COLOR = (0, 0, 255)

class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, screen_height=600):
        super().__init__()
        self.width = 20
        self.height = 20
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(AGENT_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        # Movement physics
        self.change_x = 0
        self.change_y = 0
        self.direction = 0  # -1 for left, 1 for right, 0 for no movement
        self.acceleration = 0.5
        self.friction = -0.12
        self.max_speed_x = 3
        self.screen_height = screen_height

        # Jump physics
        self.jump_speed = -7 
        self.gravity_acc = 0.4
        self.terminal_velocity = 3
        self.on_ground = False  # Initialize on_ground attribute

        self.mask = pygame.mask.from_surface(self.image)

    def update(self, blocks):
        # Apply horizontal movement
        self.accelerate()
        self.rect.x += self.change_x
        self.collide_with_blocks(self.change_x, 0, blocks)

        # Apply gravity and vertical movement
        self.apply_gravity()
        self.rect.y += self.change_y
        self.collide_with_blocks(0, self.change_y, blocks)

    def accelerate(self):
        # Apply acceleration based on direction
        self.change_x += self.direction * self.acceleration
        # Apply friction only when on the ground
        if self.on_ground:
            self.change_x += self.change_x * self.friction
        # Limit speed
        self.change_x = max(-self.max_speed_x, min(self.change_x, self.max_speed_x))


    def apply_gravity(self):
        self.change_y += self.gravity_acc
        # Limit falling speed
        if self.change_y > self.terminal_velocity:
            self.change_y = self.terminal_velocity

    def jump(self):
        if self.on_ground:
            self.change_y = self.jump_speed
            self.on_ground = False  # Set on_ground to False after jumping

    def collide_with_blocks(self, dx, dy, blocks):
        # Collision detection
        for block in blocks:
            if pygame.sprite.collide_mask(self, block):
                if isinstance(block, GoalBlock):
                    continue
                if dx > 0:
                    self.rect.right = block.rect.left
                    self.change_x = 0
                if dx < 0:
                    self.rect.left = block.rect.right
                    self.change_x = 0
                if dy > 0:
                    self.rect.bottom = block.rect.top
                    self.change_y = 0
                    self.on_ground = True  # Set on_ground to True when landing
                if dy < 0:
                    self.rect.top = block.rect.bottom
                    self.change_y = 0

    def go_left(self):
        self.direction = -1

    def go_right(self):
        self.direction = 1

    def stop(self):
        self.direction = 0