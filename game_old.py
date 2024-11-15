# game.py
import argparse
import pygame
import sys
import numpy as np
import torch
import random
import os

from agent.agent import Agent
from world.levels.level1 import Level1
from world.levels.level2 import Level2
from world.levels.level3 import Level3
from world.levels.level4 import Level4
from world.levels.level5 import Level5
from world.levels.level6 import Level6
from world.levels.level7 import Level7
from world.levels.level8 import Level8
from dqn.dqn_agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Simplified action space (movement only)
MOVEMENT_ACTIONS = {
    0: 'left',
    1: 'right',
    2: 'nothing'
}

ACTION_DIM = len(MOVEMENT_ACTIONS)  # 3 actions

# Colors
BG_COLOR = (255, 255, 255)  # Gray color

class Game:
    """
    A class to represent the game environment.

    Attributes:
    -----------
    manual_control : bool
        Flag to enable manual control of the agent.
    level_number : int
        The current level number.
    training_mode : bool
        Flag to enable training mode.
    render_enabled : bool
        Flag to enable rendering of the game.
    load_model : str or None
        Path to a pre-trained model to load.

    Methods:
    --------
    train_ai(num_episodes=1000):
        Main training loop focusing on single level mastery.
    save_training_stats():
        Save training statistics to a file.
    store_experience(state, action, reward, next_state, done):
        Store experience in replay buffer.
    train_step():
        Perform one training step.
    is_jump_necessary():
        Determine if a jump is necessary based on obstacles, traps, and gaps.
    get_state():
        Get the current state of the agent and environment.
    get_goal_distance():
        Calculate the distance to the goal.
    get_nearest_block_info():
        Get information about the nearest block in front of the agent.
    get_reward():
        Calculate the reward based on the agent's progress and actions.
    is_trap_ahead():
        Check if there is a trap ahead of the agent.
    get_nearest_trap_distance():
        Get the distance to the nearest trap.
    step(action):
        Execute one time step within the environment.
    handle_ai_action(action_index):
        Handle the AI action based on the action index.
    reset_episode():
        Reset the environment for a new episode.
    run():
        Run the game loop.
    update():
        Update the game state.
    restart_level():
        Restart the current level.
    level_completed():
        Handle the completion of a level.
    draw():
        Draw the game screen.
    quit():
        Quit the game.
    update_camera():
        Update the camera position to follow the agent.
    events():
        Handle game events.
    load_level():
        Load the current level.
    """

    def __init__(self, manual_control=False, level_number=1, training_mode=False, render_enabled=True, load_model=None):
        """
        Initialize the game environment.
        Parameters:
        manual_control (bool): If True, enables manual control of the agent. Default is False.
        level_number (int): The starting level number. Default is 1.
        training_mode (bool): If True, enables training mode for the agent. Default is False.
        render_enabled (bool): If True, enables rendering of the game. Default is True.
        load_model (str or None): Path to a pre-trained model to load. Default is None.
        Attributes:
        screen (pygame.Surface): The display surface for the game.
        clock (pygame.time.Clock): The game clock.
        running (bool): Indicates if the game is running.
        manual_control (bool): Indicates if manual control is enabled.
        level_number (int): The current level number.
        max_levels (int): The maximum number of levels.
        training_mode (bool): Indicates if training mode is enabled.
        render_enabled (bool): Indicates if rendering is enabled.
        training_active (bool): Indicates if training is active.
        agent (Agent): The agent in the game.
        all_sprites_list (pygame.sprite.Group): Group containing all sprites.
        moving_blocks (pygame.sprite.Group): Group containing moving blocks.
        block_list (list): List of blocks in the current level.
        camera_x (int): The x-offset of the camera.
        camera_y (int): The y-offset of the camera.
        training_stats (list): List to store episode results.
        current_reward (int): The current reward for the agent.
        episode_steps (int): The number of steps in the current episode.
        max_steps (int): The maximum number of steps per episode.
        last_x_position (int): The last x-position of the agent.
        last_jump_time (int): The last time the agent jumped.
        jump_cooldown (int): The cooldown period for jumps in milliseconds.
        successful_completion_times (list): List to track successful run times.
        best_completion_time (float): The best completion time.
        current_run_steps (int): The number of steps in the current run.
        last_action_index (int): The index of the last action taken.
        Training-specific attributes:
        device (torch.device): The device to run the DQN agent on.
        dqn_agent (DQNAgent): The DQN agent for training.
        replay_buffer (ReplayBuffer): The replay buffer for experience replay.
        batch_size (int): The batch size for training.
        training_steps (int): The number of training steps taken.
        target_update_freq (int): The frequency of updating the target network.
        """

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
        self.manual_control = manual_control
        self.level_number = level_number
        self.max_levels = 3
        self.training_mode = training_mode
        self.render_enabled = render_enabled
        self.training_active = False

        # Create the agent
        self.agent = Agent(50, SCREEN_HEIGHT - 80, screen_height=SCREEN_HEIGHT)
        self.load_level()

        # Sprite groups
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.level.get_all_sprites())
        self.all_sprites_list.add(self.agent)
        self.moving_blocks = pygame.sprite.Group()

        self.block_list = self.level.get_blocks()

        # Camera offset (initially zero)
        self.camera_x = 0
        self.camera_y = 0
        # Training stats
        self.training_stats = []  # Store episode results
        # Training-specific variables
        self.current_reward = 0
        self.episode_steps = 0
        self.max_steps = 5000  # Maximum steps per episode
        self.last_x_position = 50  # Track progress

        # Add jump cooldown tracking
        self.last_jump_time = 0
        self.jump_cooldown = 400  # milliseconds
        self.successful_completion_times = []  # Track only successful runs
        self.best_completion_time = float('inf')
        self.current_run_steps = 0
        self.last_action_index = 0

        if training_mode:
            # Initialize DQN components
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dim = 10  # Updated state dimension
            action_dim = ACTION_DIM  # Updated action_dim
            self.dqn_agent = DQNAgent(state_dim, action_dim, self.device)
            # Use ReplayBuffer
            self.replay_buffer = ReplayBuffer(buffer_size=50000, batch_size=64)
            self.batch_size = 64
            self.training_steps = 0
            self.target_update_freq = 1000  # Adjusted target network update frequency
            if load_model:
                self.dqn_agent.load(load_model)
                if not self.training_mode:
                    self.dqn_agent.epsilon = 0.0  # No exploration during inference

    def train_ai(self, num_episodes=1000):
        """
        Main training loop with explicit episode limit and better controls.
        """
        episode = 0
        self.training_active = True

        print(f"Starting training on level {self.level_number}")
        print("Controls:")
        print("- Press 'q' to quit training")
        print("- Press 's' to save current progress")
        print(f"Will train for {num_episodes} episodes unless stopped...")

        while self.training_active and episode < num_episodes:
            episode += 1
            self.reset_episode()
            episode_reward = 0
            self.episode_steps = 0

            while self.running and self.current_run_steps < self.max_steps:
                # Handle events - check for quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.training_active = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            print("\nStopping training...")
                            self.training_active = False
                            break
                        elif event.key == pygame.K_s:
                            print("\nManually saving progress...")
                            self.save_training_stats()
                            self.dqn_agent.save(f"checkpoint_level{self.level_number}_ep{episode}.pth")

                if not self.training_active:
                    break

                state = self.get_state()
                action = self.dqn_agent.choose_action(state)
                self.handle_ai_action(action)
                self.update()
                next_state = self.get_state()
                reward = self.get_reward()
                done = not self.running

                self.store_experience(state, action, reward, next_state, done)

                if self.replay_buffer.size() >= self.batch_size:
                    self.train_step()

                episode_reward += reward
                self.current_run_steps += 1
                self.episode_steps += 1

                if self.render_enabled:
                    self.draw()

                if done:
                    break

            # Always store the episode statistics
            episode_data = {
                'episode': episode,
                'level': self.level_number,
                'reward': float(episode_reward),  # Convert to float to ensure it's serializable
                'steps': self.current_run_steps,
                'epsilon': float(self.dqn_agent.epsilon)  # Convert to float to ensure it's serializable
            }
            self.training_stats.append(episode_data)

            # Print progress every 10 episodes
            if episode % 10 == 0:
                self.save_training_stats()  # Save more frequently
                print(f"\nEpisode {episode}/{num_episodes}:")
                print(f"  Steps: {self.current_run_steps}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Current ε: {self.dqn_agent.epsilon:.3f}")
                if self.best_completion_time != float('inf'):
                    print(f"  Best Completion Time: {self.best_completion_time}")
                print("------------------------")

            # Save model checkpoint every 100 episodes
            if episode % 100 == 0:
                self.dqn_agent.save(f"checkpoint_level{self.level_number}_ep{episode}.pth")

            self.dqn_agent.decay_epsilon()

        # Always save at the end of training
        print("\nTraining finished!")
        print(f"Completed {episode} episodes")
        self.save_training_stats()
        self.dqn_agent.save(f"final_model_level{self.level_number}.pth")

    def save_training_stats(self):
        """
        Enhanced version of save_training_stats with better error handling
        """
        if not self.training_stats:
            print("Warning: No training stats to save!")
            return

        try:
            # Save as TXT
            with open('training_results.txt', 'w') as f:
                f.write("Training Results\n")
                f.write("===============\n\n")

                # Calculate summary statistics
                total_episodes = len(self.training_stats)
                avg_reward = sum(stat['reward'] for stat in self.training_stats) / total_episodes
                avg_steps = sum(stat['steps'] for stat in self.training_stats) / total_episodes
                best_reward = max(stat['reward'] for stat in self.training_stats)

                # Write summary
                f.write(f"Summary Statistics\n")
                f.write(f"Total Episodes: {total_episodes}\n")
                f.write(f"Average Reward: {avg_reward:.2f}\n")
                f.write(f"Average Steps: {avg_steps:.2f}\n")
                f.write(f"Best Reward: {best_reward:.2f}\n")
                f.write(f"Best Completion Time: {self.best_completion_time if self.best_completion_time != float('inf') else 'N/A'}\n\n")

                # Write episode details
                f.write("Episode Details\n")
                f.write("---------------\n")
                for stat in self.training_stats:
                    f.write(f"Episode {stat['episode']}:\n")
                    f.write(f"  Level: {stat['level']}\n")
                    f.write(f"  Reward: {stat['reward']:.2f}\n")
                    f.write(f"  Steps: {stat['steps']}\n")
                    f.write(f"  Epsilon: {stat['epsilon']:.3f}\n")
                    f.write("---------------\n")

            # Save as CSV
            import pandas as pd
            df = pd.DataFrame(self.training_stats)
            df.to_csv('training_results.csv', index=False)

            print(f"Successfully saved {len(self.training_stats)} training episodes to files")

        except Exception as e:
            print(f"Error saving training stats: {e}")
            import traceback
            traceback.print_exc()

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store the experience tuple in the replay buffer.

        Args:
            state (object): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (object): The state of the environment after taking the action.
            done (bool): A flag indicating whether the episode has ended.

        Returns:
            None
        """

        self.replay_buffer.store(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one training step for the DQN agent.
        This method samples a random batch from the replay buffer and uses it to
        train the DQN agent. If the replay buffer does not have enough samples
        to form a batch, the method returns immediately. After training, the
        target network is updated periodically based on the specified frequency.

        Returns:
            None
        """

        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample random batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Train the network
        loss = self.dqn_agent.train(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self.dqn_agent.update_target_network()

    def is_jump_necessary(self):
        """
        Determines if the agent needs to jump to avoid obstacles, traps, or gaps.

        This method checks three conditions to decide if a jump is necessary:
        1. If there are obstacles directly ahead of the agent.
        2. If there are traps directly ahead of the agent.
        3. If there is no ground directly below the agent.

        Returns:
            bool: True if a jump is necessary, False otherwise.
        """
        # Define a rectangle in front of the agent
        look_ahead_rect = pygame.Rect(
            self.agent.rect.right,
            self.agent.rect.bottom - self.agent.height,
            50,  # Look ahead distance
            self.agent.height
        )

        # Check for obstacles ahead
        obstacles_ahead = any(
            look_ahead_rect.colliderect(block.rect)
            for block in self.block_list
        )

        # **Check for traps ahead**
        traps_ahead = any(
            look_ahead_rect.colliderect(trap.rect)
            for trap in self.level.trap_list
        )

        # Check for gaps (no ground below)
        ground_rect = pygame.Rect(
            self.agent.rect.x,
            self.agent.rect.bottom + 1,
            self.agent.width,
            5
        )
        ground_below = any(
            ground_rect.colliderect(block.rect)
            for block in self.block_list
        )

        # Jump is necessary if there's an obstacle, trap ahead, or no ground below
        return obstacles_ahead or traps_ahead or not ground_below

    def get_state(self):
        """
        Get the current state of the agent in the game environment.

        Returns:
            np.ndarray: A numpy array containing the following normalized state variables:
                - agent_x (float): The x-coordinate of the agent relative to the level width.
                - agent_y (float): The y-coordinate of the agent relative to the level height.
                - agent_vel_x (float): The x-velocity of the agent relative to its maximum speed.
                - agent_vel_y (float): The y-velocity of the agent relative to its terminal velocity.
                - goal_distance (float): The distance to the goal relative to the sum of the level width and height.
                - on_ground (float): 1.0 if the agent is on the ground, 0.0 otherwise.
                - obstacle_in_front (float): Information about whether there is an obstacle in front of the agent.
                - nearest_obstacle_dist (float): The distance to the nearest obstacle, normalized by 500.0.
                - trap_ahead (float): Information about whether there is a trap ahead of the agent.
                - trap_distance (float): The distance to the nearest trap, normalized by 500.0.
        """
        # Existing state variables
        agent_x = self.agent.rect.x / self.level.width
        agent_y = self.agent.rect.y / self.level.height
        agent_vel_x = self.agent.change_x / self.agent.max_speed_x
        agent_vel_y = self.agent.change_y / self.agent.terminal_velocity
        goal_distance = self.get_goal_distance() / (self.level.width + self.level.height)
        on_ground = 1.0 if self.agent.on_ground else 0.0
        nearest_obstacle_dist, obstacle_in_front = self.get_nearest_block_info()

        # **New**: Trap information
        trap_ahead = self.is_trap_ahead()
        trap_distance = self.get_nearest_trap_distance() / 500.0  # Normalize

        return np.array([
            agent_x,
            agent_y,
            agent_vel_x,
            agent_vel_y,
            goal_distance,
            on_ground,
            obstacle_in_front,
            nearest_obstacle_dist / 500.0,
            trap_ahead,
            trap_distance
        ], dtype=np.float32)

    def get_goal_distance(self):
        """
        Calculate the Euclidean distance from the agent to the nearest goal.

        This method finds the nearest goal from the level's goal list and computes
        the straight-line distance between the agent's center and the goal's center.

        Returns:
            float: The Euclidean distance to the nearest goal.
        """
        goal = next(iter(self.level.goal_list))
        distance = ((goal.rect.centerx - self.agent.rect.centerx) ** 2 +
                    (goal.rect.centery - self.agent.rect.centery) ** 2) ** 0.5
        return distance

    def get_nearest_block_info(self):
            """
            Determines the distance to the nearest block in front of the agent and whether there is an obstacle in front.

            This method creates a rectangular area in front of the agent and checks for any blocks that collide with this area.
            It then calculates the distance to the nearest block within this area.

            Returns:
                tuple: A tuple containing:
                - distance (float): The distance to the nearest block in front of the agent. If no block is found within
                  the look-ahead distance, returns 500.0.
                - obstacle_in_front (float): 1.0 if there is at least one block in front of the agent, otherwise 0.0.
            """
            look_ahead_rect = pygame.Rect(
                self.agent.rect.right,
                self.agent.rect.y,
                100,
                self.agent.height
            )
            obstacles = [block for block in self.block_list if block.rect.colliderect(look_ahead_rect)]
            obstacle_in_front = 1.0 if obstacles else 0.0
            if obstacles:
                nearest_obstacle = min(obstacles, key=lambda block: block.rect.x)
                distance = nearest_obstacle.rect.x - self.agent.rect.x
            else:
                distance = 500.0  # Max look-ahead distance

            return distance, obstacle_in_front

    def get_reward(self):
        """
        Calculate and return the reward for the agent based on its current state and actions.

        The reward is calculated based on several factors:
        - Progress towards the goal: Positive reward for moving closer to the goal, penalty for moving backward.
        - Penalty for standing still.
        - Penalty for being close to a trap: The closer the agent is to a trap, the higher the penalty.
        - Death penalties: Large penalty if the agent falls out of bounds or hits a trap.
        - Goal completion: Large reward for completing the level, with a time bonus for faster completion.

        Returns:
            float: The calculated reward for the agent.
        """
        reward = 0

        # Existing reward calculations
        current_goal_distance = self.get_goal_distance()
        distance_progress = self.last_goal_distance - current_goal_distance
        reward += distance_progress * 0.1
        self.last_goal_distance = current_goal_distance

        if distance_progress < 0:
            reward += distance_progress * 0.5  # Stronger penalty for moving backward

        # Penalty for standing still
        if self.agent.change_x == 0:
            reward -= 0.1

        # **New**: Penalize for being close to a trap
        trap_ahead = self.is_trap_ahead()
        trap_distance = self.get_nearest_trap_distance()
        if trap_ahead:
            reward -= (1 - (trap_distance / 500.0)) * 0.1  # Penalty increases as the agent gets closer to the trap

        # Death penalties
        if self.agent.rect.y > self.level.height * 2:
            reward -= 100
            self.running = False
            print(f"Failed: Fell out of bounds after {self.current_run_steps} steps")
            return reward

        # **Increased penalty for hitting a trap**
        trap_hit_list = pygame.sprite.spritecollide(self.agent, self.level.trap_list, False)
        if trap_hit_list:
            reward -= 200  # Increase penalty from -100 to -200
            self.running = False
            print(f"Failed: Hit trap after {self.current_run_steps} steps")
            return reward

        # Goal completion
        goal_hit_list = pygame.sprite.spritecollide(self.agent, self.level.goal_list, False)
        if goal_hit_list:
            if self.current_run_steps < self.best_completion_time:
                self.best_completion_time = self.current_run_steps
                print(f"\nNew best completion time: {self.best_completion_time} steps!")
                self.dqn_agent.save(f"best_model_level{self.level_number}.pth")

            time_bonus = max(0, 1000 - self.current_run_steps)
            reward += 500 + time_bonus
            self.successful_completion_times.append(self.current_run_steps)
            print(f"Level completed in {self.current_run_steps} steps!")
            avg_time = sum(self.successful_completion_times[-10:]) / len(self.successful_completion_times[-10:])
            print(f"Average completion time: {avg_time:.1f} steps")
            self.running = False

        return reward

    def is_trap_ahead(self):
        """
        Checks if there is a trap ahead of the agent.

        This method defines a rectangular area in front of the agent and checks if any traps
        are within this area. The look-ahead distance is set to 100 units.

        Returns:
            float: 1.0 if there is a trap ahead, otherwise 0.0.
        """
        # Define a rectangle in front of the agent
        look_ahead_rect = pygame.Rect(
            self.agent.rect.right,
            self.agent.rect.y,
            100,  # Look ahead distance
            self.agent.height
        )
        # Check for traps ahead
        traps_ahead = any(
            look_ahead_rect.colliderect(trap.rect)
            for trap in self.level.trap_list
        )
        return 1.0 if traps_ahead else 0.0

    def get_nearest_trap_distance(self):
        """
        Calculate the distance to the nearest trap in the agent's look-ahead path.

        This method creates a rectangular area in front of the agent to check for any traps
        within a specified maximum look-ahead distance. It then calculates the distance to
        the nearest trap within this area. If no traps are found, it returns the maximum
        look-ahead distance.

        Returns:
            float: The distance to the nearest trap. If no traps are found, returns 500.0.
        """
        look_ahead_rect = pygame.Rect(
            self.agent.rect.right,
            self.agent.rect.y,
            500,  # Max look-ahead distance
            self.agent.height
        )
        traps = [trap for trap in self.level.trap_list if trap.rect.colliderect(look_ahead_rect)]
        if traps:
            nearest_trap = min(traps, key=lambda trap: trap.rect.x)
            distance = nearest_trap.rect.x - self.agent.rect.x
        else:
            distance = 500.0  # Max look-ahead distance
        return distance

    def step(self, action):
        """
        Execute one time step within the environment.
        Parameters:
        action (int): The action to be executed in the environment.

        Returns:
        tuple: A tuple containing:
            - state (object): The current state of the environment.
            - reward (float): The reward obtained after executing the action.
            - done (bool): A flag indicating whether the episode has ended.
            - info (dict): Additional information about the environment.
        """

        if not self.running:
            return self.get_state(), 0, True, {}

        # Convert action to game commands
        self.handle_ai_action(action)

        # Update game state
        self.update()

        # Get reward and next state
        reward = self.get_reward()
        state = self.get_state()

        # Check if episode is done
        self.episode_steps += 1
        done = not self.running or self.episode_steps >= self.max_steps

        return state, reward, done, {}

    def handle_ai_action(self, action_index):
        """
        Handles the AI action based on the given action index.

        Parameters:
        action_index (int): The index of the action to be performed by the AI.
        The function retrieves the movement command corresponding to the action index
        from the MOVEMENT_ACTIONS list and performs the appropriate movement for the agent.
        It updates the last action index and handles horizontal movement commands ('left' or 'right').
        If the command is not 'left' or 'right', the agent stops moving horizontally.
        Additionally, the function checks if a jump is necessary. If a jump is required and the
        cooldown period has passed since the last jump, the agent will perform a jump action.
        """
        # Get the movement command from MOVEMENT_ACTIONS
        command = MOVEMENT_ACTIONS[action_index]
        self.last_action_index = action_index  # Store last action

        # Set horizontal movement
        if command == 'left':
            self.agent.go_left()
        elif command == 'right':
            self.agent.go_right()
        else:
            self.agent.stop()

        # The direction is maintained until the next action changes it
        # Handle jump if necessary
        if self.is_jump_necessary():
            current_time = pygame.time.get_ticks()
            if current_time - self.last_jump_time >= self.jump_cooldown:
                self.agent.jump()
                self.last_jump_time = current_time

    def reset_episode(self):
        """
        Reset the environment for a new episode.

        This method reinitializes the agent's position, movement, and various
        episode-specific variables to their starting states. It also sets the
        environment to a running state and calculates the initial distance to
        the goal.

        Returns:
            tuple: The initial state of the environment after reset.
        """

        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0
        self.running = True
        self.last_x_position = 50
        self.current_reward = 0
        self.current_run_steps = 0
        self.last_jump_time = 0
        self.last_goal_distance = self.get_goal_distance()
        return self.get_state()

    def run(self):
        """
        Runs the main game loop.

        This method will raise a ValueError if the game is attempted to be run with visualization disabled.
        The game loop will continue to run while the `self.running` flag is True, processing events, updating
        the game state, drawing the game, and limiting the frame rate to 60 FPS. Once the loop exits, the
        `self.quit()` method is called to perform any necessary cleanup.
        """
        if not self.render_enabled:
            raise ValueError("Cannot run game with visualization disabled")
        while self.running:
            self.events()
            self.update()
            self.draw()
            self.clock.tick(60)  # Limit to 60 FPS
        self.quit()

    def update(self):
        """
        Updates the game state for each frame.

        This method performs the following tasks:
        1. Updates all sprites except the agent.
        2. Updates the agent with the block list.
        3. Updates the camera position to follow the agent.
        4. Checks if the agent is out of bounds (death) and restarts the level if true.
        5. Enforces world bounds to keep the agent within the level width.
        6. Checks if the agent hits a trap and restarts the level if true.
        7. Checks if the agent reaches the goal and completes the level if true.
        """
        # Update other sprites (that don't need arguments)
        for sprite in self.all_sprites_list:
            if sprite != self.agent:
                sprite.update()

        # Update the agent with the block list
        self.agent.update(self.block_list)

        # Update camera position to follow the agent
        self.update_camera()

        # Check for out of bounds (death)
        if self.agent.rect.y > self.level.height * 2:
            self.restart_level()

        # Enforce world bounds
        if self.agent.rect.left < 0:
            self.agent.rect.left = 0
        if self.agent.rect.right > self.level.width:
            self.agent.rect.right = self.level.width

        # Check for agent hitting a trap
        trap_hit_list = pygame.sprite.spritecollide(self.agent, self.level.trap_list, False, pygame.sprite.collide_mask)
        if trap_hit_list:
            self.restart_level()

        # Check for agent reaching the goal
        goal_hit_list = pygame.sprite.spritecollide(self.agent, self.level.goal_list, False, pygame.sprite.collide_mask)
        if goal_hit_list:
            self.level_completed()

    def restart_level(self):
        """
        Restart the current level by resetting the agent's position and other necessary states.

        This method will:
        - Print a message indicating the level is restarting.
        - Reset the agent's x and y coordinates to the starting position.
        - Reset the agent's horizontal and vertical movement to zero.
        """
        print("Restarting level...")
        # Reset agent position and other necessary states
        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0

    def level_completed(self):
        """
        Handles the completion of a level in the game.
        If the game is in training mode, the function returns immediately without any action.
        Otherwise, it prints a message indicating the current level has been completed.
        If there are more levels to play, it increments the level number, prints a message
        indicating the next level is loading, and calls the load_level method to load the next level.
        If the current level is the last level, it prints a congratulatory message and sets
        the running attribute to False, indicating the game has ended.
        """
        if self.training_mode:
            return
        print(f"Level {self.level_number} Completed!")

        if self.level_number < self.max_levels:
            # Move to next level
            self.level_number += 1
            print(f"Loading level {self.level_number}...")
            self.load_level()
        else:
            print("Congratulations! You've completed all levels!")
            self.running = False

    def draw(self):
        """
        Draws the game screen.
        This method first checks if rendering is enabled. If rendering is enabled,
        it loads and scales the background image, then blits it onto the screen.
        It then iterates through all sprites in the `all_sprites_list` and blits
        each sprite onto the screen at its respective position adjusted by the
        camera offsets. Finally, it updates the display.

        Returns:
            None
        """
        if not self.render_enabled:
            return

        # self.screen.fill(BG_COLOR)
        background_image = pygame.image.load('world/assets/background-sprite.png').convert()
        background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(background_image, (0, 0))
        for entity in self.all_sprites_list:
            self.screen.blit(entity.image, (entity.rect.x + self.camera_x, entity.rect.y + self.camera_y))
        pygame.display.flip()

    def quit(self):
        """
        Quits the game by shutting down the Pygame library and exiting the program.

        This method first calls `pygame.quit()` to uninitialize all Pygame modules,
        and then calls `sys.exit()` to terminate the program.
        """
        pygame.quit()
        sys.exit()

    def update_camera(self):
        """
        Updates the camera position to center on the agent while respecting the level bounds.
        The camera position is calculated such that the agent is centered on the screen.
        The camera is then adjusted to ensure it does not show areas outside the level bounds.

        Attributes:
            self.agent.rect.x (int): The x-coordinate of the agent's rectangle.
            self.agent.rect.y (int): The y-coordinate of the agent's rectangle.
            self.level.width (int): The width of the level.
            self.level.height (int): The height of the level.
            SCREEN_WIDTH (int): The width of the screen.
            SCREEN_HEIGHT (int): The height of the screen.
            self.camera_x (int): The x-coordinate of the camera.
            self.camera_y (int): The y-coordinate of the camera.
        """
        # Calculate target camera position (centered on agent)
        target_x = -self.agent.rect.x + SCREEN_WIDTH // 2
        target_y = -self.agent.rect.y + SCREEN_HEIGHT // 2

        # Apply camera bounds
        # Left/Right bounds
        target_x = min(0, target_x)
        target_x = max(-(self.level.width - SCREEN_WIDTH), target_x)

        # Top/Bottom bounds
        target_y = min(0, target_y)  # Don't show above top of level
        target_y = max(-(self.level.height - SCREEN_HEIGHT), target_y)  # Don't show below bottom of level

        self.camera_x = target_x
        self.camera_y = target_y

    def events(self):
        """
        Handles the events in the game loop.

        This method processes events from the Pygame event queue. It handles quitting the game,
        key presses for quitting, and manual control of the agent using keyboard inputs.

        - If the event type is `pygame.QUIT`, it sets `self.running` and `self.training_active` to False.
        - If the event type is `pygame.KEYDOWN` and the key is `pygame.K_q`, it sets `self.running` and `self.training_active` to False.
        - If `self.manual_control` is True and `self.training_mode` is False, it allows manual control of the agent:
            - `pygame.K_LEFT` key makes the agent go left.
            - `pygame.K_RIGHT` key makes the agent go right.
            - No left or right key press makes the agent stop.
            - `pygame.K_SPACE` key makes the agent jump.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.training_active = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                    self.training_active = False

        if self.manual_control and not self.training_mode:
            # Manual control using keyboard inputs
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.agent.go_left()
            elif keys[pygame.K_RIGHT]:
                self.agent.go_right()
            else:
                self.agent.stop()
            if keys[pygame.K_SPACE]:
                self.agent.jump()

    def load_level(self):
        """
        Loads the current level based on the level number.

        This method initializes the level by selecting the appropriate level class
        from a predefined dictionary of level classes. If the level number does not
        match any key in the dictionary, it defaults to Level1. It then creates an
        instance of the selected level class and updates the sprite groups with the
        sprites from the level and the agent.

        Attributes:
            level_classes (dict): A dictionary mapping level numbers to level classes.
            level_class (class): The class of the level to be loaded.
            level (object): An instance of the selected level class.
            all_sprites_list (pygame.sprite.Group): A group containing all sprites in the level.
            block_list (pygame.sprite.Group): A group containing all block sprites in the level.
        """
        # You can expand this dictionary as you add more levels
        level_classes = {
            1: Level1,
            2: Level2,
            3: Level3,
            4: Level4,
            5: Level5,
            6: Level6,
            7: Level7,
            8: Level8
            # Add more levels here
        }

        level_class = level_classes.get(self.level_number, Level1)
        self.level = level_class()

        # Update sprite groups
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.level.get_all_sprites())
        self.all_sprites_list.add(self.agent)
        self.block_list = self.level.get_blocks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent World")
    parser.add_argument('--m', action='store_true', help='Enable manual control for testing.')
    parser.add_argument('--l', type=int, default=1, help='Select level number to load.')
    parser.add_argument('--t', action='store_true', help='Enable AI training mode.')
    parser.add_argument('--r', action='store_true', help='Enable visualization during training.')
    parser.add_argument('--lm', type=str, help='Load Model: Path to the trained model file.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')

    args = parser.parse_args()

    game = Game(
        manual_control=args.m,
        level_number=args.l,
        training_mode=args.t,
        render_enabled=args.r if args.t else True,
        load_model=args.lm
    )

    if args.t:
        game.train_ai(num_episodes=args.episodes)
    else:
        game.run()
