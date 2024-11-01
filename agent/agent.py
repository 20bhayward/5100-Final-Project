# agent.py
import numpy as np
import pygame
from world.components.blocks.interactive.goal_block import GoalBlock

AGENT_COLOR = (0, 0, 255)

class Agent(pygame.sprite.Sprite):


    """
    A class to represent an agent in a game.

    Attributes
    ----------
    width : int
        The width of the agent.
    height : int
        The height of the agent.
    image : pygame.Surface
        The surface representing the agent.
    rect : pygame.Rect
        The rectangle representing the agent's position and size.
    change_x : float
        The horizontal speed of the agent.
    change_y : float
        The vertical speed of the agent.
    direction : int
        The direction of the agent's movement (-1 for left, 1 for right, 0 for no movement).
    acceleration : float
        The acceleration of the agent.
    friction : float
        The friction applied to the agent's movement.
    max_speed_x : float
        The maximum horizontal speed of the agent.
    screen_height : int
        The height of the game screen.
    jump_speed : float
        The speed at which the agent jumps.
    gravity_acc : float
        The acceleration due to gravity.
    terminal_velocity : float
        The maximum falling speed of the agent.
    on_ground : bool
        Whether the agent is on the ground.
    mask : pygame.Mask
        The mask for pixel-perfect collision detection.

    Methods
    -------
    update(blocks):
        Updates the agent's position and handles collisions with blocks.
    accelerate():
        Applies acceleration and friction to the agent's horizontal movement.
    apply_gravity():
        Applies gravity to the agent's vertical movement.
    jump():
        Makes the agent jump if it is on the ground.
    collide_with_blocks(dx, dy, blocks):
        Handles collisions with blocks.
    go_left():
        Sets the agent's direction to left.
    go_right():
        Sets the agent's direction to right.
    stop():
        Stops the agent's horizontal movement.
    """
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