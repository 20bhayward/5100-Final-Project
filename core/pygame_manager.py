# pygame_manager.py
import pygame
from pygame import Color
import os
import numpy as np
from .config import SCREEN_WIDTH, SCREEN_HEIGHT
import trainer.trainer

class PygameManager:
    def __init__(self, render_enabled=True, game=None, agent=None):
        """Initialize pygame and display first."""
        self.render_enabled = render_enabled
        self.game = game
        self.level = None
        self.trainer = None

        # Set video driver before pygame.init()
        if not render_enabled:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()

        # IMPORTANT: Set display mode immediately after pygame.init()
        if render_enabled:
            self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        else:
            self.screen = pygame.Surface([SCREEN_WIDTH, SCREEN_HEIGHT])
            pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

        self.camera_x = 0
        self.camera_y = 0
        self.camera_shift = 0

        self.running = True
        self.clock = pygame.time.Clock()

        # Initialize sprite groups
        self.all_sprites_list = pygame.sprite.Group()
        self.moving_blocks = pygame.sprite.Group()
        self.blocks = pygame.sprite.Group()

        self.agent = agent
        if agent:
            self.all_sprites_list.add(agent)

    def set_agent(self, agent):
        """Safely set or update the agent."""
        if self.agent:
            self.all_sprites_list.remove(self.agent)
        self.agent = agent
        if agent:
            self.all_sprites_list.add(agent)

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

        if self.game.manual_control:  # Manual control mode
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.agent.go_left()
            elif keys[pygame.K_RIGHT]:
                self.agent.go_right()
            else:
                self.agent.stop()  # Important: stop when no movement keys are pressed

            if keys[pygame.K_SPACE]:
                if self.agent.on_ground and not self.agent.is_jumping:
                    self.agent.jump()
        else:  # AI control mode - make sure agent doesn't move unless commanded
            if not hasattr(self, 'trainer') or not self.trainer.training_mode:
                self.agent.stop()  # Stop agent if not in training mode

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
        """Lock the frame rate to the specified FPS."""
        pygame.time.Clock().tick(fps)

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

        # Load and draw the background
        background_image = pygame.image.load('world/assets/background-sprite.png').convert()
        background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(background_image, (0, 0))

        # Draw all sprites with camera adjustment
        for entity in self.all_sprites_list:
            self.screen.blit(entity.image, (entity.rect.x + self.camera_x, entity.rect.y + self.camera_y))

        self.draw_agent_percepts()

        pygame.display.flip()



    def draw_agent_percepts(self):
        """Visualize the agent's percepts."""
        agent = self.agent
        precepts = self.game.trainer.precepts
        percepts_data = precepts.get_visible_state()

        # Adjust positions based on camera
        agent_screen_x = agent.rect.x + self.camera_x
        agent_screen_y = agent.rect.y + self.camera_y

        # Draw current platform in green
        if percepts_data['platforms']['current_platform']:
            platform_rect = percepts_data['platforms']['current_platform']['rect']
            rect = pygame.Rect(platform_rect.x + self.camera_x, platform_rect.y + self.camera_y,
                            platform_rect.width, platform_rect.height)
            pygame.draw.rect(self.screen, Color('green'), rect, 2)

        # Draw gaps
        for gap in percepts_data['platforms']['gaps']:
            gap_start_x = gap['start_x'] + self.camera_x
            gap_end_x = gap_start_x + gap['width']
            y = agent.rect.bottom + self.camera_y
            color = Color('red') if not gap['jumpable'] else Color('yellow')
            pygame.draw.line(self.screen, color, (gap_start_x, y), (gap_end_x, y), 3)

        # Draw hazards
        for threat in percepts_data['hazards']['immediate_threats']:
            hazard_rect = threat['rect']
            rect = pygame.Rect(hazard_rect.x + self.camera_x, hazard_rect.y + self.camera_y,
                            hazard_rect.width, hazard_rect.height)
            pygame.draw.rect(self.screen, Color('red'), rect, 2)

        # Draw goal direction
        goal_direction = percepts_data['goal_direction']
        if goal_direction != 0:
            arrow_start = (agent.rect.centerx + self.camera_x, agent.rect.top + self.camera_y - 20)
            arrow_end = (arrow_start[0] + 50 * goal_direction, arrow_start[1])
            pygame.draw.line(self.screen, Color('magenta'), arrow_start, arrow_end, 2)
            # Draw arrowhead
            pygame.draw.polygon(self.screen, Color('magenta'), [
                (arrow_end[0], arrow_end[1]),
                (arrow_end[0] - 5 * goal_direction, arrow_end[1] - 5),
                (arrow_end[0] - 5 * goal_direction, arrow_end[1] + 5)
            ])

        # Display movement state
        font = pygame.font.SysFont(None, 24)
        movement = percepts_data['movement']
        movement_text = f"Vel X: {movement['velocity_x']:.2f}, Vel Y: {movement['velocity_y']:.2f}, On Ground: {movement['on_ground']}"
        movement_surface = font.render(movement_text, True, Color('white'))
        self.screen.blit(movement_surface, (10, SCREEN_HEIGHT - 30))

    def clear_sprites(self):
        """Clear all sprite groups"""
        self.all_sprites_list.empty()
        self.moving_blocks.empty()
        self.block_list = []

    def quit(self):
        pygame.quit()
