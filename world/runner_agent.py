import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
AGENT_SIZE = 50
OBSTACLE_WIDTH = 30
OBSTACLE_HEIGHT = 60
GROUND_LEVEL = SCREEN_HEIGHT - AGENT_SIZE
SPEED = 5
JUMP_STRENGTH = 15
GRAVITY = 1

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Runner")

# Agent class
class Agent:
    def __init__(self):
        self.rect = pygame.Rect(100, GROUND_LEVEL, AGENT_SIZE, AGENT_SIZE)
        self.velocity_y = 0
        self.is_jumping = False

    def move(self, action):
        if action == 0:  # Move left
            self.rect.x -= SPEED
        elif action == 1:  # Move right
            self.rect.x += SPEED
        elif action == 2 and not self.is_jumping:  # Jump
            self.is_jumping = True
            self.velocity_y = -JUMP_STRENGTH

    def apply_gravity(self):
        if self.is_jumping:
            self.rect.y += self.velocity_y
            self.velocity_y += GRAVITY
            if self.rect.y >= GROUND_LEVEL:
                self.rect.y = GROUND_LEVEL
                self.is_jumping = False

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, self.rect)

# Obstacle class
class Obstacle:
    def __init__(self):
        self.rect = pygame.Rect(SCREEN_WIDTH, GROUND_LEVEL - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

    def update(self):
        self.rect.x -= SPEED

    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)

def main():
    clock = pygame.time.Clock()
    agent = Agent()
    obstacles = []
    running = True
    score = 0

    # Game loop
    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get agent action (0: left, 1: right, 2: jump)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1
        elif keys[pygame.K_SPACE]:
            action = 2
        else:
            action = -1

        # Move the agent
        if action != -1:
            agent.move(action)

        # Apply gravity
        agent.apply_gravity()

        # Randomly spawn obstacles
        if random.randint(1, 100) < 5:  # Small chance of spawning an obstacle
            obstacles.append(Obstacle())

        # Update obstacles
        for obstacle in obstacles[:]:
            obstacle.update()
            if obstacle.rect.right < 0:
                obstacles.remove(obstacle)
                score += 1  # Score increases as obstacles are passed

        # Check for collisions
        for obstacle in obstacles:
            if agent.rect.colliderect(obstacle.rect):
                running = False  # End the game on collision

        # Draw agent and obstacles
        agent.draw(screen)
        for obstacle in obstacles:
            obstacle.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    print(f"Game Over! Score: {score}")
    pygame.quit()

if __name__ == "__main__":
    main()
