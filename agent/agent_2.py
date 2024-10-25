import pygame

# Constants (consider moving these to a config file or passing as arguments)
SCREEN_HEIGHT = 400
SCREEN_WIDTH = 800
AGENT_SIZE = 30
GROUND_LEVEL = SCREEN_HEIGHT - AGENT_SIZE
SPEED = 5
JUMP_STRENGTH = 17
GRAVITY = 1
BLACK = (0, 0, 0)

class Agent:
    def __init__(self):
        # Agent's properties such as position, velocity, etc.
        self.rect = pygame.Rect(100, GROUND_LEVEL, AGENT_SIZE, AGENT_SIZE)
        self.velocity_x = 0  # Horizontal velocity
        self.velocity_y = 0  # Vertical velocity
        self.is_jumping = False
        self.jump_strength = JUMP_STRENGTH
        self.gravity = GRAVITY
        self.screen_width = SCREEN_WIDTH  # Define screen bounds
        self.safe_distance = 150  # Distance to prepare for a jump

    def move(self, action):
        """
        Handle movement actions.
        action: 0 = move left, 1 = move right, 2 = jump
        """
        if action == 0:  # Move left
            self.velocity_x = -SPEED
        elif action == 1:  # Move right
            self.velocity_x = SPEED
        elif action == 2 and not self.is_jumping:  # Jump
            self.is_jumping = True
            self.velocity_y = -self.jump_strength  # Start the jump by giving negative velocity

    def intelligent_move(self, obstacles, action):
        """
        Intelligent movement considering obstacles.
        action: 0 = move left, 1 = move right, 2 = jump
        """
        # Call the move method based on action input
        self.move(action)

        # # Sort obstacles by their x-position (only obstacles ahead of the agent)
        # obstacles_ahead = [o for o in obstacles if o.rect.x > self.rect.x]
        # obstacles_ahead.sort(key=lambda o: o.rect.x)

        # # Calculate distances to obstacles and react
        # distances_to_obstacles = [(obstacle, obstacle.rect.x - self.rect.x) for obstacle in obstacles_ahead]

        # # Ideal jumping distance window (adjust these values for your game mechanics)
        # min_jump_distance = 10
        # max_jump_distance = 40

        # # Check for jump conditions when action is 2 (jump)
        # if action == 2 and not self.is_jumping:
        #     for i, (obstacle, distance_to_obstacle) in enumerate(distances_to_obstacles):
        #         if min_jump_distance < distance_to_obstacle < max_jump_distance:
        #             print(f"Jumping over obstacle {i+1}")
        #             self.is_jumping = True
        #             self.velocity_y = -self.jump_strength
        #             break

        # Apply gravity
        self.apply_gravity()

        # Update horizontal position
        self.rect.x += self.velocity_x
        self._constrain_agent_within_bounds()
        self.velocity_x = 0

    def apply_gravity(self):
        """Apply gravity to the agent if it's in the air."""
        self.velocity_y += self.gravity
        self.rect.y += self.velocity_y

        # Check if the agent has landed on the ground
        if self.rect.y >= GROUND_LEVEL:
            self.rect.y = GROUND_LEVEL
            self.is_jumping = False
            self.velocity_y = 0  # Stop falling

    def _constrain_agent_within_bounds(self):
        """
        Ensure that the agent stays within the screen bounds.
        Prevents the agent from moving out of the left or right edges of the screen.
        """
        # Ensure the agent doesn't go out of the left or right bounds
        if self.rect.x < 0:
            self.rect.x = 0  # Constrain to the left edge
        elif self.rect.x > self.screen_width - self.rect.width:
            self.rect.x = self.screen_width - self.rect.width  # Constrain to the right edge

    def update(self, obstacles, action):
        """
        Update the agent's position intelligently based on obstacles and the selected action.
        """
        self.intelligent_move(obstacles, action)

    def draw(self, screen):
        """Draw the agent on the screen."""
        pygame.draw.rect(screen, BLACK, self.rect)
