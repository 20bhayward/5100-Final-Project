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
    def __init__(self):
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
        self.agent = Agent()  # Reset agent
        self.obstacles = []  # Reset obstacles
        self.score = 0
        self.done = False
        return self._get_obs()

    def step(self, action):
        if self.done:
            return self.reset()

        # Check if the agent has fallen out of the screen
        if self.agent.rect.center[0] < 0 or self.agent.rect.center[0] > 0.7 * SCREEN_WIDTH:
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

        self.agent.apply_gravity(self.obstacles)

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
        Calculate the reward based on the agent's actions and environment state.

        - Penalize the agent heavily for collisions with obstacles.
        - Reward the agent for jumping over obstacles.
        - Penalize slightly for unnecessary jumps.
        - Reward left/right movement if it brings the agent closer to an ideal jump position.
        """
        if done:
            return -20  # Higher penalty for hitting an obstacle

        nearest_obstacle = None
        if self.obstacles:
            nearest_obstacle = min(self.obstacles, key=lambda o: o.rect.x)
            distance_to_obstacle = nearest_obstacle.rect.x - self.agent.rect.x

            # Ideal jumping distance window (tune these values for your game mechanics)
            min_jump_distance = 20
            max_jump_distance = 60

            # Reward for jumping within the ideal window
            if  min_jump_distance < distance_to_obstacle < max_jump_distance and self.agent.is_jumping:
                return 10  # High reward for jumping within the ideal window

            # Reward for moving right if too far from the obstacle
            if distance_to_obstacle > max_jump_distance:
                if self.agent.velocity_x > 0:  # Moving right toward the obstacle
                    return 2  # Reward for moving closer to the obstacle

            # Reward for moving left if too close to the obstacle (within 20 pixels)
            if 0 < distance_to_obstacle < min_jump_distance and self.agent.velocity_x < 0:
                return 2  # Reward for backing up to better position for the jump

            # Penalize slightly for unnecessary jumping
            if self.agent.is_jumping and (distance_to_obstacle > max_jump_distance or distance_to_obstacle <= 0):
                return -2  # Small penalty for jumping unnecessarily

        # Default reward for moving forward and passing obstacles
        return 1

    def render(self, mode='human'):
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
        pygame.quit()

    def _get_obs(self):
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
