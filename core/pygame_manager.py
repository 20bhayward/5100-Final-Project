# pygame_manager.py
import pygame
import os
from .config import SCREEN_WIDTH, SCREEN_HEIGHT

class PygameManager:
    def __init__(self, render_enabled=True, game=None, agent=None):
        self.game = game
        self.agent = agent
        self.render_enabled = render_enabled
        self.level = None  # Will be set by Game class
        self.trainer = None  # Will be set by Game class

        # Initialize pygame with the dummy video driver if render is disabled
        if not render_enabled:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()

        # Set up the display
        if render_enabled:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("AI Agent World")
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Initialize pygame display even if we're not rendering
        if not pygame.display.get_surface():
            pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.HIDDEN)

        self.clock = pygame.time.Clock()
        self.running = True

        # Sprite groups
        self.all_sprites_list = pygame.sprite.Group()
        self.moving_blocks = pygame.sprite.Group()
        self.block_list = []

        # Camera offset (initially zero)
        self.camera_x = 0
        self.camera_y = 0

    def get_screen(self):
        return self.screen

    def event_handler(self):
        """Handle pygame events and manual control"""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                if hasattr(self.trainer, 'training_active'):
                    self.trainer.training_active = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                    if hasattr(self.trainer, 'training_active'):
                        self.trainer.training_active = False

        if self.game.manual_control:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.agent.go_left()
            elif keys[pygame.K_RIGHT]:
                self.agent.go_right()
            else:
                self.agent.stop()
            if keys[pygame.K_SPACE]:
                self.agent.jump()
        return events

    def update(self, level, agent):
        """
        Updates the game state for each frame.
        """
        # Update other sprites (that don't need arguments)
        for sprite in self.get_all_sprites():
            if sprite != agent:
                sprite.update()

        # Update the agent with the block list
        agent.update(self.get_blocks())

        # Update camera position to follow the agent
        self.update_camera(agent, level.width, level.height)

        # Check for out of bounds (death)
        if agent.rect.y > level.height * 2:
            self.game.restart_level()

        # Enforce world bounds
        if agent.rect.left < 0:
            agent.rect.left = 0
        if agent.rect.right > level.width:
            agent.rect.right = level.width

        # Check for agent hitting a trap
        trap_hit_list = pygame.sprite.spritecollide(agent, level.trap_list, False, pygame.sprite.collide_mask)
        if trap_hit_list:
            self.game.restart_level()

        # Check for agent reaching the goal
        goal_hit_list = pygame.sprite.spritecollide(agent, level.goal_list, False, pygame.sprite.collide_mask)
        if goal_hit_list:
            self.game.level_completed()

    def tick(self, fps=60):
        self.clock.tick(fps)

    def is_running(self):
        return self.running

    def set_running(self, running):
        self.running = running

    def add_sprites(self, sprites):
        self.all_sprites_list.add(sprites)

    def add_moving_blocks(self, blocks):
        self.moving_blocks.add(blocks)

    def set_blocks(self, blocks):
        self.block_list = blocks

    def get_all_sprites(self):
        return self.all_sprites_list

    def get_blocks(self):
        return self.block_list

    def update_camera(self, agent, level_width, level_height):
        """Update camera position to follow agent"""
        # Calculate target camera position (centered on agent)
        target_x = -agent.rect.x + SCREEN_WIDTH // 2
        target_y = -agent.rect.y + SCREEN_HEIGHT // 2

        # Apply camera bounds
        target_x = min(0, target_x)  # Don't show past left edge
        target_x = max(-(level_width - SCREEN_WIDTH), target_x)  # Don't show past right edge

        target_y = min(0, target_y)  # Don't show above top of level
        target_y = max(-(level_height - SCREEN_HEIGHT), target_y)  # Don't show below bottom of level

        self.camera_x = target_x
        self.camera_y = target_y

    def draw(self, all_sprites_list, background_image_path):
        """Draw the game state"""
        if not self.render_enabled:
            return

        background_image = pygame.image.load('world/assets/background-sprite.png').convert()
        background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(background_image, (0, 0))

        for entity in self.all_sprites_list:
            self.screen.blit(entity.image, (entity.rect.x + self.camera_x, entity.rect.y + self.camera_y))

        pygame.display.flip()

    def clear_sprites(self):
        """Clear all sprite groups"""
        self.all_sprites_list.empty()
        self.moving_blocks.empty()
        self.block_list = []

    def quit(self):
        pygame.quit()
