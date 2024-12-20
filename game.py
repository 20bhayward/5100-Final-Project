import pygame
from agent.agent import Agent
from core.geneticAlgorithm import genetic_algorithm
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.static.square_block import SquareBlock
import random

pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("f(AI) Dash")

# Game parameters
scroll_speed = 3  # How fast the screen scrolls
camera_offset = 0  # Tracks the camera's horizontal scroll offset

# Platform setup
platform = pygame.sprite.Group()
PLATFORM_Y = SCREEN_HEIGHT - 50  # Fixed height for the platform
PLATFORM_BLOCK_SIZE = 40         # Size of each platform block
PLATFORM_LENGTH = SCREEN_WIDTH // PLATFORM_BLOCK_SIZE + 2  # Blocks to fill the screen

# Generate platform
for i in range(PLATFORM_LENGTH):
    block = SquareBlock(x=i * PLATFORM_BLOCK_SIZE, y=PLATFORM_Y, size=PLATFORM_BLOCK_SIZE)
    platform.add(block)

# Agent setup
agent = Agent(50, PLATFORM_Y - 40, screen_height=SCREEN_HEIGHT)  # Place agent on top of the platform
all_sprites = pygame.sprite.Group(agent)

# Obstacle generation
OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 30
OBSTACLE_Y = PLATFORM_Y - OBSTACLE_HEIGHT  # Place obstacles on the platform

def generate_cyclic_obstacles(initial_distance, obstacle_group, obstacle_y, obstacle_width, obstacle_height):
    distances = [300, 350, 400]  # Cycle distances for obstacle spacing
    distance_index = 0
    obstacles_to_generate = 1000
    obstacles_per_distance = 5

    current_position = initial_distance
    for i in range(obstacles_to_generate):
        trap = TrapBlock(
            x=current_position,
            y=obstacle_y,
            width=obstacle_width,
            height=obstacle_height
        )
        obstacle_group.add(trap)
        current_position += distances[distance_index]

        # Cycle through the distances
        if (i + 1) % obstacles_per_distance == 0:
            distance_index = (distance_index + 1) % len(distances)

    return current_position

obstacles = pygame.sprite.Group()
last_position = generate_cyclic_obstacles(
    initial_distance=100,
    obstacle_group=obstacles,
    obstacle_y=OBSTACLE_Y,
    obstacle_width=OBSTACLE_WIDTH,
    obstacle_height=OBSTACLE_HEIGHT
)
# Train AI using a genetic algorithm
best_dna, _ = genetic_algorithm(agent, generations=500, population_size=500, num_actions=2000, obstacles=obstacles)

# Main game loop
running = True
clock = pygame.time.Clock()
# Play the best solution
action_index = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((135, 206, 250))  # Sky-blue background

    # Scroll platform
    for block in platform:
        block.rect.x -= scroll_speed
        if block.rect.right < 0:
            platform.remove(block)

    # Dynamically add blocks to platform
    last_block = max(block.rect.right for block in platform)
    if last_block < SCREEN_WIDTH:
        platform.add(SquareBlock(x=last_block, y=PLATFORM_Y, size=PLATFORM_BLOCK_SIZE))

    platform.draw(screen)

    # Scroll obstacles
    for obstacle in obstacles:
        obstacle.rect.x -= scroll_speed
    
    # Remove off-screen obstacles
    obstacles = pygame.sprite.Group(obstacle for obstacle in obstacles if obstacle.rect.right > 0)

    # Dynamically generate new obstacles
    if len(obstacles) < 10:  # Ensure enough traps on-screen
        last_position = generate_cyclic_obstacles(
            initial_distance=last_position,
            obstacle_group=obstacles,
            obstacle_y=OBSTACLE_Y,
            obstacle_width=OBSTACLE_WIDTH,
            obstacle_height=OBSTACLE_HEIGHT
        )

    obstacles.draw(screen)

    # Execute the AI's actions
    if action_index < len(best_dna):
        action = best_dna[action_index]
        if action == "jump":
            agent.jump()
        action_index += 1
    else:
        print("Replay finished")
        running = False  # Stop the game when all actions are executed

    # Update and draw the agent
    collision_blocks = pygame.sprite.Group(platform, obstacles)
    all_sprites.update(collision_blocks)
    all_sprites.draw(screen)

    # Display the agent's current distance
    font = pygame.font.Font(None, 36)
    distance_text = font.render(f"Distance: {agent.rect.x}px", True, (0, 0, 0))
    screen.blit(distance_text, (10, 10))

    # Update the display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()