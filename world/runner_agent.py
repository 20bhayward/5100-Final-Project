import pygame
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent_2 import Agent
from components.obstacle import Obstacle

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
JUMP_STRENGTH = 20
GRAVITY = 1

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Runner")

# Agent class

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
        agent.apply_gravity(obstacles)
        agent.update_position()

        # Randomly spawn obstacles
        if random.randint(1, 100) < 3:  # Small chance of spawning an obstacle
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
