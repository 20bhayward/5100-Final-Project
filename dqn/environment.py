import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym
from gym import spaces
import numpy as np
import pygame
import random
from agent.agent import Agent
from world.components.obstacle import Block


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BG_COLOR = (50, 50, 50)

# Define the custom environment
class PlatformEnv(gym.Env):
    def __init__(self):
        super(PlatformEnv, self).__init__()
        self.screen_width = 800
        self.screen_height = 600
        self.agent_width = 40
        self.agent_height = 40
        self.jump_height = 40
        self.action_space = spaces.Discrete(4)  # Left, Right, Jump, Do Nothing
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("AI Agent World")
        self.reset()

    def reset(self):
        # Initialize the agent at the starting position
        self.agent = Agent(50, self.screen_height - 80)
        # Generate the platforms
        self.generate_platforms()
        # Set the done flag to False
        self.done = False
        # Return the initial state
        return self._get_state()

    def generate_platforms(self):
        # Create sprite groups for blocks and all sprites
        self.block_list = pygame.sprite.Group()
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.agent)
        platform_width = 40
        platform_height = 40
        max_platform_height = 5
        min_platform_height = 1
        max_gap = 300
        min_gap = 100
        x = 0
        last_platform_top = self.screen_height
        while x < self.screen_width * 3:
            num_blocks_high = random.randint(min_platform_height, max_platform_height)
            platform_top = self.screen_height - num_blocks_high * platform_height
            if last_platform_top - platform_top > self.agent.jump_height:
                platform_top = last_platform_top - self.agent.jump_height
            if x == 0:
                platform_top = self.screen_height - platform_height
            for h in range(num_blocks_high):
                y = self.screen_height - (h + 1) * platform_height
                for i in range(0, platform_width * random.randint(1, 3), platform_width):
                    block = Block(x + i, y)
                    self.block_list.add(block)
                    self.all_sprites_list.add(block)
            last_platform_top = platform_top
            x += platform_width * random.randint(1, 3) + random.randint(min_gap, max_gap)

    def _get_state(self):
        # Capture the current screen state as an array
        state = pygame.surfarray.array3d(pygame.display.get_surface())
        return state

    def step(self, action):
        # Update the agent's position based on the action
        if action == 0:  # Left
            self.agent.rect.x -= 10
        elif action == 1:  # Right
            self.agent.rect.x += 10
        elif action == 2:  # Jump
            self.agent.rect.y -= self.agent.jump_height
        elif action == 3:  # Do Nothing
            pass
        # Apply gravity
        self.agent.rect.y += 10
        # Initialize the reward
        reward = -1
        # Check if the agent has fallen below the screen
        if self.agent.rect.y > self.screen_height:
            self.done = True
            reward = -100
        # Return the current state, reward, and done flag
        return self._get_state(), reward, self.done, {}

    def render(self, mode='human'):
        # Fill the screen with the background color
        self.screen.fill(BG_COLOR)
        # Draw all sprites on the screen
        self.all_sprites_list.draw(self.screen)
        # Update the display
        pygame.display.flip()

    def close(self):
        # Quit Pygame
        pygame.quit()
