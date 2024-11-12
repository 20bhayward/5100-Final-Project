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
    """
    Custom environment that follows the OpenAI Gym interface.
    This environment simulates a platformer game where an agent can move left, right, jump, or do nothing.

    Attributes:
        screen_width (int): Width of the game screen.
        screen_height (int): Height of the game screen.
        agent_width (int): Width of the agent.
        agent_height (int): Height of the agent.
        jump_height (int): Height the agent can jump.
        action_space (gym.spaces.Discrete): The action space, consisting of 4 actions (Left, Right, Jump, Do Nothing).
        observation_space (gym.spaces.Box): The observation space, representing the screen as an array.
        screen (pygame.Surface): The game screen.
        done (bool): Flag indicating if the episode is done.
        agent (Agent): The agent in the environment.
        block_list (pygame.sprite.Group): Group of platform blocks.
        all_sprites_list (pygame.sprite.Group): Group of all sprites in the environment.

    Methods:
        __init__(): Initializes the environment.
        reset(): Resets the environment to the initial state.
        generate_platforms(): Generates platforms for the agent to navigate.
        _get_state(): Captures the current screen state as an array.
        step(action): Updates the agent's position based on the action and returns the current state, reward, and done flag.
        render(mode='human'): Renders the environment.
        close(): Closes the environment and quits Pygame.
    """
    def __init__(self):
        """
        Initializes the PlatformEnv environment.

        This sets up the screen dimensions, agent dimensions, jump height, action space, 
        and observation space for the environment. It also initializes the Pygame library 
        and sets up the display window.

        Attributes:
            screen_width (int): The width of the game screen.
            screen_height (int): The height of the game screen.
            agent_width (int): The width of the agent.
            agent_height (int): The height of the agent.
            jump_height (int): The height the agent can jump.
            action_space (gym.spaces.Discrete): The action space for the agent, with 4 possible actions (Left, Right, Jump, Do Nothing).
            observation_space (gym.spaces.Box): The observation space representing the screen as an array of pixel values.
            screen (pygame.Surface): The Pygame display surface.
        """
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
        """
        Resets the environment to its initial state.

        This method performs the following actions:
        1. Initializes the agent at the starting position.
        2. Generates the platforms in the environment.
        3. Sets the done flag to False.

        Returns:
            state (object): The initial state of the environment.
        """
        # Initialize the agent at the starting position
        self.agent = Agent(50, self.screen_height - 80)
        # Generate the platforms
        self.generate_platforms()
        # Set the done flag to False
        self.done = False
        # Return the initial state
        return self._get_state()

    def generate_platforms(self):
        """
        Generates platforms for the game environment.

        This method creates sprite groups for blocks and all sprites, including the agent.
        It generates platforms of varying heights and gaps, ensuring that the agent can
        jump between them. The platforms are created by adding blocks to the block list
        and the all sprites list.

        Attributes:
            block_list (pygame.sprite.Group): Group containing all block sprites.
            all_sprites_list (pygame.sprite.Group): Group containing all sprites, including the agent.
            platform_width (int): Width of each platform block.
            platform_height (int): Height of each platform block.
            max_platform_height (int): Maximum number of blocks in platform height.
            min_platform_height (int): Minimum number of blocks in platform height.
            max_gap (int): Maximum gap between platforms.
            min_gap (int): Minimum gap between platforms.
            x (int): Current x position for platform generation.
            last_platform_top (int): Top position of the last generated platform.
        """
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
        """
        Capture the current screen state as a 3D array.

        This method uses the pygame library to capture the current state of the display surface
        and returns it as a 3D array representing the RGB values of each pixel.

        Returns:
            numpy.ndarray: A 3D array representing the current screen state.
        """
        # Capture the current screen state as an array
        state = pygame.surfarray.array3d(pygame.display.get_surface())
        return state

    def step(self, action):
        """
        Execute one time step within the environment based on the given action.

        Args:
            action (int): The action to be taken by the agent. 
                          0 - Move Left
                          1 - Move Right
                          2 - Jump
                          3 - Do Nothing

        Returns:
            tuple: A tuple containing:
                - state (object): The current state of the environment.
                - reward (int): The reward received after taking the action.
                - done (bool): A flag indicating whether the episode has ended.
                - info (dict): Additional information (empty dictionary in this case).
        """
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
        """
        Renders the current state of the environment to the screen.

        Args:
            mode (str): The mode in which to render the environment. Default is 'human'.

        This method fills the screen with the background color, draws all sprites on the screen,
        and updates the display.
        """
        # Fill the screen with the background color
        self.screen.fill(BG_COLOR)
        # Draw all sprites on the screen
        self.all_sprites_list.draw(self.screen)
        # Update the display
        pygame.display.flip()

    def close(self):
        """
        Closes the environment and quits Pygame.

        This method should be called to properly shut down the environment and 
        release any resources used by Pygame.
        """
        # Quit Pygame
        pygame.quit()
