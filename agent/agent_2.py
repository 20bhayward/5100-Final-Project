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
    def __init__(self):
        self.rect = pygame.Rect(100, GROUND_LEVEL, AGENT_SIZE, AGENT_SIZE)
        self.velocity_x = 0  # Horizontal velocity
        self.velocity_y = 0  # Vertical velocity
        self.is_jumping = False
        self.on_moving_platform = None  # Track a moving platform the agent might be on

    def move(self, action):
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

    def check_on_moving_platform(self, platforms):
        """
        Check if the agent is on a moving platform 
        and update position accordingly.
        """
        self.on_moving_platform = None  # Reset the reference each frame
        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                print(f"Agent is colliding with platform at {platform.rect.topleft}")
                self.on_moving_platform = platform
                break  # Stop checking if on a platform


    def is_falling_on_top(self, obstacle):
        """
        Check if the agent is falling onto the top of the obstacle.
        This prevents landing on the side or bottom of the obstacle.
        """
        # Ensure the agent is above the obstacle and falling downwards
        if (self.rect.bottom <= obstacle.rect.top + 10 and  # Agent is above obstacle
            self.velocity_y >= 0 and  # Agent is falling down
            obstacle.rect.left < self.rect.right and  # Agent is horizontally aligned with obstacle
            obstacle.rect.right > self.rect.left):  # Agent is horizontally aligned with obstacle
            return True
        return False

    def update_position(self):
        # Update the horizontal position based on velocity
        self.rect.x += self.velocity_x

        # Check if on a moving platform and update position based on its movement
        if self.on_moving_platform:
            print("Inside moving block")
            self.rect.x += self.on_moving_platform.speed  # Adjust according to platform speed

        # Reset horizontal velocity after movement
        self.velocity_x = 0

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, self.rect)
