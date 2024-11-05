import pygame

# Constants (consider moving these to a config file or passing as arguments)
SCREEN_HEIGHT = 400
AGENT_SIZE = 30
GROUND_LEVEL = SCREEN_HEIGHT - AGENT_SIZE
SPEED = 5
JUMP_STRENGTH = 17
GRAVITY = 1
BLACK = (0, 0, 0)

class Agent:
    """
    A class to represent an agent in a game.

    Attributes
    ----------
    rect : pygame.Rect
        The rectangle representing the agent's position and size.
    velocity_x : int
        The horizontal velocity of the agent.
    velocity_y : int
        The vertical velocity of the agent.
    is_jumping : bool
        A flag indicating whether the agent is currently jumping.

    Methods
    -------
    __init__():
        Initializes the agent with default position and velocities.
    move(action):
        Moves the agent based on the given action.
    apply_gravity(obstacles):
        Applies gravity to the agent and checks for collisions with obstacles.
    is_falling_on_top(obstacle):
        Checks if the agent is falling onto the top of an obstacle.
    update_position():
        Updates the agent's position based on its velocity.
    draw(screen):
        Draws the agent on the given screen.
    """
        
    def __init__(self):
        """
        Initializes the agent with default position and velocities.
        
        Parameters
        ----------
        None
        """
        self.rect = pygame.Rect(100, GROUND_LEVEL, AGENT_SIZE, AGENT_SIZE)
        self.velocity_x = 0  # Horizontal velocity
        self.velocity_y = 0  # Vertical velocity
        self.is_jumping = False

    def move(self, action):
        """
        Controls the movement of the agent based on the given action.

        Parameters:
        action (int): The action to be performed by the agent.
                      0 - Move left
                      1 - Move right
                      2 - Jump (if not already jumping)

        Returns:
        None
        """
        if action == 0:  # Move left
            if not self.is_jumping:
                self.velocity_x = -SPEED
        elif action == 1:  # Move right
            if not self.is_jumping:
                self.velocity_x = SPEED
        elif action == 2 and not self.is_jumping:  # Jump
            self.is_jumping = True
            self.velocity_y = -JUMP_STRENGTH

    def apply_gravity(self, obstacles):
        """
        Applies gravity to the agent and checks for collisions with the ground and obstacles.

        Args:
            obstacles (list): A list of obstacle objects that the agent can collide with.

        Updates:
            self.velocity_y (float): The vertical velocity of the agent, increased by gravity.
            self.rect.y (int): The vertical position of the agent, updated by the vertical velocity.
            self.is_jumping (bool): Set to False if the agent lands on the ground or an obstacle.
        """
        # Apply gravity to vertical velocity
        self.velocity_y += GRAVITY
        self.rect.y += self.velocity_y
        # Check if the agent is landing on the ground
        if self.rect.y >= GROUND_LEVEL:
            self.rect.y = GROUND_LEVEL
            self.is_jumping = False
            self.velocity_y = 0  # Stop falling
        # Check for collision with any obstacles
        for obstacle in obstacles:
            if self.rect.colliderect(obstacle.rect):
                self.rect.y = obstacle.rect.top - self.rect.height  # Position agent on top of obstacle
                self.is_jumping = False
                self.velocity_y = 0  # Stop falling
                break  # Stop checking once landed

    def is_falling_on_top(self, obstacle):
        """
        Determine if the agent is falling on top of the given obstacle.

        Args:
            obstacle: An object with a 'rect' attribute that represents its position and size.

        Returns:
            bool: True if the agent is above the obstacle, falling downwards, and horizontally aligned with the obstacle; False otherwise.
        """
        # Ensure the agent is above the obstacle and falling downwards
        if (self.rect.bottom <= obstacle.rect.top + 10 and  # Agent is above obstacle
            self.velocity_y >= 0 and  # Agent is falling down
            obstacle.rect.left < self.rect.right and  # Agent is horizontally aligned with obstacle
            obstacle.rect.right > self.rect.left):  # Agent is horizontally aligned with obstacle
            return True
        return False

    def update_position(self):
        """
        Updates the position of the agent.

        This method updates the horizontal position of the agent based on its current
        horizontal velocity (`velocity_x`). After updating the position, the horizontal
        velocity is reset to 0.
        """
        # Update the horizontal position based on velocity
        self.rect.x += self.velocity_x

        # Reset horizontal velocity after movement
        self.velocity_x = 0

    def draw(self, screen):
        """
        Draws the agent on the given screen.

        Args:
            screen (pygame.Surface): The surface on which to draw the agent.
        """
        pygame.draw.rect(screen, BLACK, self.rect)
