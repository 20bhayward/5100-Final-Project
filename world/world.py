import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
BG_COLOR = (50, 50, 50)
AGENT_COLOR = (0, 255, 0)
BLOCK_COLOR = (255, 0, 0)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Agent World")

# Clock to control frame rate
clock = pygame.time.Clock()

# Agent class
class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.width = 20
        self.height = 20
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(AGENT_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        # Velocity
        self.change_x = 0
        self.change_y = 0

    def update(self, blocks):
        # Apply gravity
        self.gravity()

        # Move left/right
        self.rect.x += self.change_x

        # Check for collision with blocks
        block_hit_list = pygame.sprite.spritecollide(self, blocks, False)
        for block in block_hit_list:
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                self.rect.left = block.rect.right

        # Move up/down
        self.rect.y += self.change_y

        # Check for collision with blocks
        block_hit_list = pygame.sprite.spritecollide(self, blocks, False)
        for block in block_hit_list:
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
                self.change_y = 0
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom
                self.change_y = 0

    def gravity(self):
        if self.change_y == 0:
            self.change_y = 1
        else:
            self.change_y += 0.35  # Adjust gravity here

    def jump(self, blocks):
        # Move down a bit to see if we are on the ground
        self.rect.y += 2
        block_hit_list = pygame.sprite.spritecollide(self, blocks, False)
        self.rect.y -= 2

        # If it is ok to jump, set our speed upwards
        if len(block_hit_list) > 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.change_y = -10  # Adjust jump strength here

    def go_left(self):
        self.change_x = -6  # Adjust movement speed here

    def go_right(self):
        self.change_x = 6

    def stop(self):
        self.change_x = 0

# Block class
class Block(pygame.sprite.Sprite):
    def __init__(self, x, y, width=40, height=40):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(BLOCK_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

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
