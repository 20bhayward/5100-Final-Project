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
        """
        Initializes the Agent object with specified position and screen height.

        Args:
            x (int): The initial x-coordinate of the agent.
            y (int): The initial y-coordinate of the agent.
            screen_height (int, optional): The height of the screen. Defaults to 600.

        Attributes:
            width (int): The width of the agent.
            height (int): The height of the agent.
            image (pygame.Surface): The surface representing the agent.
            rect (pygame.Rect): The rectangle representing the agent's position and size.
            change_x (float): The change in x-coordinate for movement.
            change_y (float): The change in y-coordinate for movement.
            direction (int): The direction of movement (-1 for left, 1 for right, 0 for no movement).
            acceleration (float): The acceleration rate of the agent.
            friction (float): The friction applied to the agent's movement.
            max_speed_x (float): The maximum horizontal speed of the agent.
            screen_height (int): The height of the screen.
            jump_speed (float): The initial speed when the agent jumps.
            gravity_acc (float): The acceleration due to gravity.
            terminal_velocity (float): The maximum falling speed of the agent.
            on_ground (bool): Whether the agent is on the ground.
            mask (pygame.mask.Mask): The mask for pixel-perfect collision detection.
        """
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
        """
        Updates the agent's position based on its current velocity and checks for collisions with blocks.

        Args:
            blocks (list): A list of block objects that the agent can collide with.

        The method performs the following steps:
        1. Applies horizontal movement by updating the agent's x-coordinate based on its horizontal velocity.
        2. Checks for collisions with blocks after horizontal movement.
        3. Applies gravity to affect the agent's vertical velocity.
        4. Applies vertical movement by updating the agent's y-coordinate based on its vertical velocity.
        5. Checks for collisions with blocks after vertical movement.
        """
        # Apply horizontal movement
        self.accelerate()
        self.rect.x += self.change_x
        self.collide_with_blocks(self.change_x, 0, blocks)

        # Apply gravity and vertical movement
        self.apply_gravity()
        self.rect.y += self.change_y
        self.collide_with_blocks(0, self.change_y, blocks)

    def accelerate(self):
        """
        Accelerates the agent by modifying its horizontal velocity.

        This method adjusts the agent's horizontal velocity (`change_x`) based on its
        current direction and acceleration. If the agent is on the ground, friction
        is also applied to the horizontal velocity. The horizontal velocity is then
        limited to ensure it does not exceed the maximum allowed speed.

        Attributes:
            direction (float): The current direction of the agent (positive for right, negative for left).
            acceleration (float): The rate at which the agent accelerates.
            on_ground (bool): Whether the agent is currently on the ground.
            friction (float): The friction coefficient applied when the agent is on the ground.
            max_speed_x (float): The maximum horizontal speed the agent can achieve.
            change_x (float): The current horizontal velocity of the agent.
        """
        # Apply acceleration based on direction
        self.change_x += self.direction * self.acceleration
        # Apply friction only when on the ground
        if self.on_ground:
            self.change_x += self.change_x * self.friction
        # Limit speed
        self.change_x = max(-self.max_speed_x, min(self.change_x, self.max_speed_x))


    def apply_gravity(self):
        """
        Applies gravity to the agent by increasing the vertical change in position (change_y) 
        by the gravity acceleration (gravity_acc). If the resulting change_y exceeds the 
        terminal velocity, it is capped at the terminal velocity to limit the falling speed.
        """
        self.change_y += self.gravity_acc
        # Limit falling speed
        if self.change_y > self.terminal_velocity:
            self.change_y = self.terminal_velocity

    def jump(self):
        """
        Makes the agent jump by setting its vertical change speed to the jump speed
        and marking it as not on the ground.

        This method should be called when the agent needs to jump. It will only
        have an effect if the agent is currently on the ground.
        """
        if self.on_ground:
            self.change_y = self.jump_speed
            self.on_ground = False  # Set on_ground to False after jumping

    def collide_with_blocks(self, dx, dy, blocks):
        """
        Handles collision detection and response with blocks.

        Args:
            dx (int): The change in the x-direction.
            dy (int): The change in the y-direction.
            blocks (list): A list of block sprites to check for collisions.

        Notes:
            - If a collision is detected with a block that is an instance of GoalBlock, the collision is ignored.
            - If dx > 0, the agent is moving right and its right side is aligned with the left side of the block.
            - If dx < 0, the agent is moving left and its left side is aligned with the right side of the block.
            - If dy > 0, the agent is moving down and its bottom side is aligned with the top side of the block.
            - If dy < 0, the agent is moving up and its top side is aligned with the bottom side of the block.
            - When the agent lands on a block (dy > 0), the on_ground attribute is set to True.
        """
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
        """
        Sets the agent's direction to left.

        This method updates the agent's direction attribute to -1, 
        indicating that the agent should move to the left.
        """
        self.direction = -1

    def go_right(self):
        """
        Sets the agent's direction to right.

        This method updates the agent's direction attribute to 1, indicating that the agent should move to the right.
        """
        self.direction = 1

    def stop(self):
        """
        Stops the agent by setting its direction to 0.
        """
        self.direction = 0