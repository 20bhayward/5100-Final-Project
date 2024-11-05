import gym
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym import spaces
import pygame
import numpy as np
from agent.agent_2 import Agent
from game_logic import spawn_obstacle, update_obstacles, check_collisions

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
WHITE = (255, 255, 255)

class RunnerEnv(gym.Env):
    """
    RunnerEnv is a custom OpenAI Gym environment for a 2D runner game using Pygame.

    Attributes:
        action_space (gym.spaces.Discrete): Action space representing 5 discrete actions:
            0: Move left
            1: Move right
            2: Jump
            3: Jump + Move Left
            4: Jump + Move Right
        observation_space (gym.spaces.Box): Observation space representing the state of the environment.
        screen_width (int): Width of the game screen.
        screen_height (int): Height of the game screen.
        screen (pygame.Surface): Pygame display surface.
        clock (pygame.time.Clock): Pygame clock to control the frame rate.
        agent (Agent): The agent controlled by the player.
        obstacles (list): List of obstacle objects in the game.
        score (int): The current score of the player.
        done (bool): Flag indicating whether the current episode is done.

    Methods:
        __init__(): Initializes the environment and its attributes.
        reset(): Resets the environment to its initial state.
        step(action): Executes a step in the environment based on the given action.
        _calculate_reward(done): Calculates the reward based on the agent's actions and environment state.
        render(mode='human'): Renders the game screen.
        close(): Closes the Pygame window.
        _get_obs(): Returns the current observation of the environment.
    """
        
    def __init__(self):
        """
        Initialize the RunnerEnv environment.

        This method sets up the action space, observation space, and initializes
        the game elements including the screen, agent, and obstacles.

        Action space:
            - Discrete(5): 
                0: move left
                1: move right
                2: jump
                3: jump + move left
                4: jump + move right

        Observation space:
            - Box(low, high, dtype=np.float32): 
                - Agent x position
                - Agent y position
                - Agent velocity
                - Nearest obstacle x position
                - Nearest obstacle y position
                - Nearest obstacle height

        Initializes:
            - Pygame and screen with specified width and height.
            - Agent object.
            - List of obstacles.
            - Score counter.
            - Done flag to indicate if the game is over.
            - Pygame clock for controlling the frame rate.
        """
        super(RunnerEnv, self).__init__()

        # Define action space and observation space
        self.action_space = spaces.Discrete(5)  # 0: left, 1: right, 2: jump, 3: jump + left, 4: jump + right
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),  # Agent x, y, velocity, nearest obstacle x, nearest obstacle y, nearest obstacle height
            high=np.array([SCREEN_WIDTH, SCREEN_HEIGHT, 100, 100, SCREEN_WIDTH, SCREEN_HEIGHT, 100]),
            dtype=np.float32
        )

        # Initialize pygame and screen
        pygame.init()
        self.screen_width = 800
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("2D Runner")
        self.clock = pygame.time.Clock()

        # Initialize game elements
        self.agent = Agent()
        self.obstacles = []
        self.score = 0
        self.done = False
        self.clock = pygame.time.Clock()

    def reset(self):
        """
        Resets the environment to its initial state.

        This method resets the agent, clears the obstacles, sets the score to 0,
        and marks the environment as not done. It returns the initial observation
        of the environment.

        Returns:
            object: The initial observation of the environment.
        """
        self.agent = Agent()  # Reset agent
        self.obstacles = []  # Reset obstacles
        self.score = 0
        self.done = False
        return self._get_obs()

    def step(self, action):
        """
        Execute one time step within the environment.

        Parameters:
        action (int): The action to be taken by the agent. 
                      0 - Move left
                      1 - Move right
                      2 - Jump
                      3 - Jump + Move Left
                      4 - Jump + Move Right

        Returns:
        tuple: A tuple containing:
            - observation (object): The current observation of the environment.
            - reward (float): The reward obtained from taking the action.
            - done (bool): Whether the episode has ended.
            - info (dict): Additional information about the environment.
        """
        if self.done:
            return self.reset()  # Consider separating logic here

        # Check if the agent has fallen out of the screen
        if self.agent.rect.center[0] < 0 or self.agent.rect.center[0] > 0.5 * SCREEN_WIDTH:
            self.done = True  # Set done to True, terminate the episode
            return self._get_obs(), -10, True, {}  # Penalty for going out of bounds

        # Move the agent and apply gravity based on the action
        if action == 0:  # Move left
            self.agent.move(0)
        elif action == 1:  # Move right
            self.agent.move(1)
        elif action == 2:  # Jump
            self.agent.move(2)
        elif action == 3:  # Jump + Move Left
            self.agent.move(2)
            self.agent.move(0)
        elif action == 4:  # Jump + Move Right
            self.agent.move(2)
            self.agent.move(1)

        self.agent.apply_gravity(self.obstacles)  # Ensure gravity is applied correctly
        self.agent.update_position()

        # Spawn and update obstacles
        spawn_obstacle(self.obstacles)
        update_obstacles(self.obstacles)

        # Update obstacles and score
        for obstacle in self.obstacles[:]:
            obstacle.update()
            if obstacle.rect.right < 0:
                self.obstacles.remove(obstacle)
                self.score += 1  # Score increases as obstacles are passed

        # Check for collisions
        done = check_collisions(self.agent, self.obstacles)

        # Assign rewards
        reward = self._calculate_reward(done)

        return self._get_obs(), reward, done, {}

    def _calculate_reward(self, done):
        """
        Calculate the reward for the agent based on its current state and actions.

        Parameters:
        done (bool): A flag indicating whether the episode is done (e.g., the agent hit an obstacle).

        Returns:
        int: The calculated reward based on the agent's actions and proximity to obstacles.

        Reward Structure:
        - If the episode is done (agent hit an obstacle), return a penalty of -20.
        - If the agent successfully jumps over an obstacle, return a reward of 50.
        - If the agent jumps within the ideal distance window (20 < distance < 40), return a reward of 1.
        - If the agent moves closer to an obstacle when too far (distance > 40), return a reward of 1.
        - If the agent backs up when too close to an obstacle (0 < distance < 20), return a reward of 2.
        - If the agent jumps unnecessarily (distance > 40), return a small penalty of -5.
        - Otherwise, return a neutral reward of 0.
        """

        if done:
            return -20  # Higher penalty for hitting an obstacle

        nearest_obstacle = None
        if self.obstacles:
            nearest_obstacle = min(self.obstacles, key=lambda o: o.rect.x)
            obstacles_ahead = [o for o in self.obstacles if o.rect.x > self.agent.rect.x]
            distance_to_obstacle = 0
            if obstacles_ahead:
                nearest_obstacle_ahead = min(obstacles_ahead, key=lambda o: o.rect.x)
                distance_to_obstacle = nearest_obstacle_ahead.rect.x - self.agent.rect.x

            # Ideal jumping distance window (tune these values for your game mechanics)
            min_jump_distance = 20
            max_jump_distance = 40

            # Reward for jumping within the ideal window
            if min_jump_distance < distance_to_obstacle < max_jump_distance and self.agent.is_jumping:
                return 1  # High reward for jumping within the ideal window

            if self.agent.rect.x > nearest_obstacle.rect.x and self.agent.rect.y >= nearest_obstacle.rect.y:
                print("jumped over an obstacle")
                return 50  # Reward for successfully jumping over the obstacle

            # Reward for moving closer to the obstacle if too far
            if distance_to_obstacle > max_jump_distance and self.agent.velocity_x > 0:
                return 1  # Reward for moving closer to the obstacle

            # Reward for backing up if too close
            if 0 < distance_to_obstacle < min_jump_distance and self.agent.velocity_x < 0:
                return 2  # Reward for backing up to better position for the jump

            # Penalize slightly for unnecessary jumping
            if self.agent.is_jumping and distance_to_obstacle > max_jump_distance:
                return -5  # Small penalty for jumping unnecessarily

        return 0

    def render(self, mode='human'):
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode in which to render the environment. Default is 'human'.

        This method performs the following steps:
        1. Fills the screen with a white background.
        2. Draws the agent on the screen.
        3. Draws all obstacles on the screen.
        4. Flips the display to update the rendered content.
        5. Updates the display.
        6. Handles events such as window closing.
        7. Maintains a frame rate of 60 frames per second.
        """
        self.screen.fill(WHITE)
        self.agent.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        pygame.display.flip()
         # Update display
        pygame.display.update()
        # Handle events (like resizing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        pygame.event.get()
        self.clock.tick(60)  # 60 FPS



    def close(self):
        """
        Shuts down the Pygame library and closes the game window.
        """
        pygame.quit()

    def _get_obs(self):
        """
        Get the current observation of the environment.

        This method returns the state of the environment as a numpy array, which includes:
        - The agent's x-coordinate.
        - The agent's y-coordinate.
        - The agent's velocity in the x direction.
        - The agent's velocity in the y direction.
        - The distance to the nearest obstacle in the x direction.
        - The y-coordinate of the nearest obstacle.
        - The height of the nearest obstacle.

        If there are no obstacles, the distance to the nearest obstacle is set to the screen width,
        the y-coordinate of the nearest obstacle is set to the screen height, and the height of the
        nearest obstacle is set to 0.

        Returns:
            np.ndarray: A numpy array containing the agent's position, velocity, and information
                        about the nearest obstacle.
        """
        nearest_obstacle = None
        if self.obstacles:
            nearest_obstacle = min(self.obstacles, key=lambda o: o.rect.x)
            distance_to_obstacle = nearest_obstacle.rect.x - self.agent.rect.x
            obstacle_y = nearest_obstacle.rect.y
            obstacle_height = nearest_obstacle.rect.height
        else:
            distance_to_obstacle = SCREEN_WIDTH
            obstacle_y = SCREEN_HEIGHT
            obstacle_height = 0

        # Return the agent's x, y, velocity, nearest obstacle x, y, and obstacle height
        return np.array([
            self.agent.rect.x,
            self.agent.rect.y,
            self.agent.velocity_x,
            self.agent.velocity_y,
            distance_to_obstacle,
            obstacle_y,
            obstacle_height
        ], dtype=np.float32)

# Testing the environment
# if __name__ == "__main__":
#     env = RunnerEnv()
#     for episode in range(5):
#         state = env.reset()
#         done = False
#         score = 0
#         while not done:
#             env.render()
#             action = env.action_space.sample()  # Replace with your RL agent's action
#             state, reward, done, _ = env.step(action)
#             score += reward
#         print(f"Episode {episode+1}: Score: {score}")
#     env.close()
