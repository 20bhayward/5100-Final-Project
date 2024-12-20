# agent.py
import numpy as np
import pygame
from PIL import Image
# from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.trap_block import TrapBlock

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
        self.height = 40
        self.width = 40
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(AGENT_COLOR)  # Fallback: solid blue square
        self.rect = self.image.get_rect(topleft=(x, y))
        self.y = y
        # self.frames = self.load_gif_frames("world/assets/runner-sprite.gif")
        # self.current_frame = 0  # Start on the first frame
        # self.image = self.frames[self.current_frame]
        # self.rect = self.image.get_rect(topleft=(x, 589))
        self.animation_speed = 5  # Number of game cycles per frame
        self.frame_counter = 0

        # Movement physics
        self.change_x = 2
        self.change_y = 0
        # self.gravity_acc = 0.4
        # self.direction = 0  # -1 for left, 1 for right, 0 for no movement
        self.acceleration = 0.5
        # self.friction = -0.12
        # self.max_speed_x = 3
        # self.screen_height = screen_height

        # Jump physics
        self.gravity_acc = 0.4
        self.terminal_velocity = 3
        self.jump_speed = -8
        self.on_ground = True  # Initialize on_ground attribute

        self.mask = pygame.mask.from_surface(self.image)

    def load_gif_frames(self, gif_path):
        """Load frames from a GIF and convert them to Pygame surfaces."""
        frames = []
        gif = Image.open(gif_path)
        for frame in range(gif.n_frames):
            gif.seek(frame)
            frame_image = gif.convert("RGBA")
            frame_surface = pygame.image.fromstring(frame_image.tobytes(), frame_image.size, frame_image.mode)
            frames.append(frame_surface)
        return frames

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
        # self.accelerate()
        # self.rect.x += self.change_x
        # self.collide_with_blocks(self.change_x, 0, blocks)

        # # Apply gravity and vertical movement
        # self.apply_gravity()
        # self.rect.y += self.change_y
        # self.collide_with_blocks(0, self.change_y, blocks)
    
        # self.animate()
        
        self.rect.x += self.change_x

        # Apply gravity
        self.apply_gravity()
        self.rect.y += self.change_y

        # Check collisions
        self.collide_with_blocks(0, self.change_y, blocks)

        if self.on_ground:
            self.change_y = 0

        # Animate the agent
        self.animate()

    def animate(self):
        """Animate by cycling through GIF frames."""
        self.frame_counter += 1
        if self.frame_counter >= self.animation_speed:
            # self.current_frame = (self.current_frame + 1) % len(self.frames)
            # self.image = self.frames[self.current_frame]
            self.mask = pygame.mask.from_surface(self.image)
            self.frame_counter = 0

    def apply_gravity(self):
        """
        Applies gravity to the agent by increasing the vertical change in position (change_y) 
        by the gravity acceleration (gravity_acc). If the resulting change_y exceeds the 
        terminal velocity, it is capped at the terminal velocity to limit the falling speed.
        """
        if not self.on_ground:
            self.change_y += self.gravity_acc
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
            # print(f"Block position: {block.rect}, Agent position: {self.rect}")  # Debug block positions
            if pygame.sprite.collide_mask(self, block):

                if isinstance(block, TrapBlock):
                    print("Agent died!")  # Debug message
                    self.die()  # Call a method to handle death
                    return
                
                print("hello", block)  # Debug collision detection
                if dy > 0:  # Falling down
                    self.rect.bottom = block.rect.top
                    self.change_y = 0
                    self.on_ground = True  # Set on_ground to True when landing
                elif dy < 0:  # Jumping up
                    self.rect.top = block.rect.bottom
                    self.change_y = 0
                if dx > 0:  # Moving right
                    self.rect.right = block.rect.left
                    self.change_x = 0
                elif dx < 0:  # Moving left
                    self.rect.left = block.rect.right
                    self.change_x = 0
    def die(self):
        # Reset the agent's position or trigger game-over logic
        print("Game Over! The agent touched a trap.")  # Debug message
        self.reset()

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
    
    def test_obstacle_sequence(self, sequence):
        """
        Simulates the agent interacting with a given sequence of obstacles.

        Args:
            sequence (list): A list of obstacles (each represented as a dictionary).

        Returns:
            bool: True if the agent can successfully "complete" the sequence, False otherwise.
        """
        # Simulate testing by evaluating the sequence
        for obstacle in sequence:
            obstacle_type = obstacle.get("type", "unknown")
            height = obstacle.get("height", 0)

            # Example logic: the agent fails if the obstacle height exceeds a threshold
            if obstacle_type == "spike" and height > 50:
                return False
        return True

    def reset(self):
        """Reset the agent's position, velocity, and state."""
        self.rect.x = 50  # Reset to the starting x position
        self.rect.y = self.y  # Reset above the platform
        self.change_y = 0  # Stop vertical movement
        self.on_ground = False