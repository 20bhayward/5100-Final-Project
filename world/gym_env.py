import gym
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym import spaces
import pygame
import random
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
        self.action_space = spaces.Discrete(3)  # 0: left, 1: right, 2: jump

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -100, -100, 0, 0]),  # Agent x, y, velocity_x, velocity_y, distance to nearest obstacle ahead, distance to nearest obstacle behind
            high=np.array([SCREEN_WIDTH, SCREEN_HEIGHT, 100, 100, SCREEN_WIDTH, SCREEN_WIDTH]),  # Max values for each
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

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self, 'action_space'):
            self.action_space.seed(seed)
        if hasattr(self, 'observation_space'):
            self.observation_space.seed(seed)

    def reset(self):
        self.agent = Agent()  # Reset agent
        self.obstacles = []  # Reset obstacles
        self.score = 0
        self.done = False
        return self._get_obs()

    def step(self, action):
        # print(self.agent.rect.y)
        if self.done:
            return self.reset()  # Consider separating logic here

        # Check if the agent has fallen out of the screen horizontally
        if self.agent.rect.center[0] < 0 or self.agent.rect.center[0] > 0.5 * SCREEN_WIDTH:
            self.done = True  # Set done to True, terminate the episode
            return self._get_obs(), -1000, True, {}  # Penalty for going out of bounds

        # Intelligent movement based on multiple obstacles
        self.agent.intelligent_move(self.obstacles, action)

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
        reward = 0

        # High penalty if the agent collides with an obstacle
        if done:
            reward -= 100

        # Identify obstacles ahead and behind
        obstacles_ahead = [o for o in self.obstacles if o.rect.x > self.agent.rect.x]
        obstacles_behind = [o for o in self.obstacles if o.rect.x <= self.agent.rect.x]

        # Sort obstacles by distance
        obstacles_ahead.sort(key=lambda o: o.rect.x)
        obstacles_behind.sort(key=lambda o: o.rect.x, reverse=True)

        # Get the nearest obstacles
        obstacle_ahead = obstacles_ahead[0] if obstacles_ahead else None
        obstacle_behind = obstacles_behind[0] if obstacles_behind else None

        # Reward for optimal jump over obstacle ahead
        if obstacle_ahead:
            distance_to_obstacle_ahead = obstacle_ahead.rect.x - self.agent.rect.x

            # Reward for jumping at the right distance
            if self.agent.is_jumping:
                if 15 <= distance_to_obstacle_ahead <= 40:
                    print("jumped at the right time")
                    reward += 20  # Optimal timing
                elif distance_to_obstacle_ahead < 5:
                    print("jumped too late")
                    reward -= 10  # Penalty for jumping too late
                    print("jumped too early")
                elif distance_to_obstacle_ahead > 40:
                    reward -= 1  # Penalty for jumping too early
            elif not self.agent.is_jumping and distance_to_obstacle_ahead > 50:
                print("not jumping when no obstacle nearby")
                reward += 1

        # Penalty for jumping with no obstacles in range
        if self.agent.is_jumping and not obstacle_ahead and not obstacle_behind:
            print("jumping for no reason")
            reward -= 1

        # Encourage forward motion away from close obstacles behind
        if obstacle_behind:
            distance_to_obstacle_behind = self.agent.rect.x - obstacle_behind.rect.x
            if distance_to_obstacle_behind > 15:
                print("keeping a good distance from the last object behind")
                reward += 5  # Small reward for keeping distance from obstacle behind
            elif distance_to_obstacle_behind <= 5:
                print("too close to the object behind")
                reward -= 10  # Penalty if too close to an obstacle behind

        # New: Reward/Penalty for Adjusting X-Position (Better Jump/Land Positioning)
        if self.agent.is_jumping:
            # Encourage the agent to aim for a landing zone between obstacles if in the air
            if obstacle_ahead and obstacle_behind:
                gap_between_obstacles = obstacle_ahead.rect.x - obstacle_behind.rect.x
                distance_to_next_obstacle = distance_to_obstacle_ahead

                if gap_between_obstacles > 20:  # Sufficient space to land
                    if 15 <= distance_to_next_obstacle <= 40:
                        reward += 10  # Reward for optimal x-position while in the air
                    else:
                        reward -= 5  # Penalty for suboptimal position while in the air
        else:
            # If the agent is on the ground, encourage movement towards the next obstacle for better jump position
            if obstacle_ahead:
                distance_to_obstacle_ahead = obstacle_ahead.rect.x - self.agent.rect.x
                if 10 <= distance_to_obstacle_ahead <= 40:
                    reward += 5  # Reward for positioning well on the ground
                elif distance_to_obstacle_ahead < 15:
                    reward -= 5  # Penalty for being too close without jumping

        # Default to zero if no reward or penalty conditions are met
        return reward if reward != 0 else 0

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
        distance_to_obstacle_ahead = None
        distance_to_obstacle_behind = None
        if self.obstacles:
            obstacles_ahead = [o for o in self.obstacles if o.rect.x > self.agent.rect.x]
            obstacles_behind = [o for o in self.obstacles if o.rect.x <= self.agent.rect.x]

            obstacles_ahead.sort(key=lambda o: o.rect.x)
            obstacles_behind.sort(key=lambda o: o.rect.x, reverse=True)

            obstacle_ahead = obstacles_ahead[0] if len(obstacles_ahead) > 0 else None
            obstacle_behind = obstacles_behind[0] if len(obstacles_behind) > 0 else None

            distance_to_obstacle_ahead = obstacle_ahead.rect.x - self.agent.rect.x if obstacle_ahead else None
            distance_to_obstacle_behind = np.abs(obstacle_behind.rect.x - self.agent.rect.x) if obstacle_behind else None
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
            distance_to_obstacle_ahead,
            distance_to_obstacle_behind
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
