import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import Agent
from components.obstacle import Block


# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
BG_COLOR = (50, 50, 50)
BLOCK_COLOR = (255, 0, 0)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Agent World")

# Clock to control frame rate
clock = pygame.time.Clock()

# Create sprite groups
all_sprites_list = pygame.sprite.Group()
block_list = pygame.sprite.Group()

# Create blocks/platforms
for i in range(0, SCREEN_WIDTH, 40):
    block = Block(i, SCREEN_HEIGHT - 40)
    block_list.add(block)
    all_sprites_list.add(block)

# Add some blocks in the middle
block = Block(200, 400)
block_list.add(block)
all_sprites_list.add(block)

block = Block(400, 300)
block_list.add(block)
all_sprites_list.add(block)

# Create the agent
agent = Agent(50, SCREEN_HEIGHT - 80)
all_sprites_list.add(agent)

# Camera offset
camera_x = 0
camera_y = 0

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Controls for interaction
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                agent.go_left()
            if event.key == pygame.K_RIGHT:
                agent.go_right()
            if event.key == pygame.K_SPACE:
                agent.jump(block_list)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT and agent.change_x < 0:
                agent.stop()
            if event.key == pygame.K_RIGHT and agent.change_x > 0:
                agent.stop()

    # Update the agent
    agent.update(block_list)

    # Update camera to follow the agent
    camera_x = -agent.rect.x + SCREEN_WIDTH // 2
    camera_y = -agent.rect.y + SCREEN_HEIGHT // 2

    # Draw everything
    screen.fill(BG_COLOR)

    for entity in all_sprites_list:
        screen.blit(entity.image, (entity.rect.x + camera_x, entity.rect.y + camera_y))

    # Update the screen
    pygame.display.flip()

    # Limit to 60 frames per second
    clock.tick(60)

pygame.quit()
sys.exit()
