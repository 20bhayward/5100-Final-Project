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
from world.dqn.dqn_agent import DQNAgent
from world.dqn.replay_buffer import ReplayBuffer

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
    def __init__(self, manual_control=False, level_number=1, training_mode=False, render_enabled=True, load_model=None):
        if not render_enabled:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

        if render_enabled:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("AI Agent World")
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            
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
        """Main training loop focusing on single level mastery"""
        episode = 0
        self.training_active = True
        
        print(f"Starting training on level {self.level_number}")
        print("Press 'q' at any time to stop training...")

        while self.training_active:
            episode += 1
            self.reset_episode()
            episode_reward = 0
            self.episode_steps = 0  # Reset episode steps

            while self.running and self.current_run_steps < self.max_steps:
                self.events()

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

                if done or not self.training_active:
                    break  # Exit the inner loop if training is no longer active

            # Print progress
            print(f"Episode {episode}:")
            print(f"  Steps: {self.current_run_steps}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Current ε: {self.dqn_agent.epsilon:.3f}")
            if self.best_completion_time != float('inf'):
                print(f"  Best Completion Time: {self.best_completion_time}")
            print("------------------------")

            # Save progress periodically
            if episode % 100 == 0:
                self.dqn_agent.save(f"checkpoint_level{self.level_number}_ep{episode}.pth")
                self.save_training_stats()

            self.dqn_agent.decay_epsilon()
            self.clock.tick(60)
            # Check if training was stopped
            if not self.training_active:
                print("Training terminated by user.")
                self.dqn_agent.save(f"checkpoint_level{self.level_number}_ep{episode}.pth")  # Save progress
                self.save_training_stats()
                pygame.quit()
                sys.exit()

    def save_training_stats(self):
        """Save training statistics to a file"""
        with open('training_results.txt', 'w') as f:
            f.write("Training Results\n")
            f.write("===============\n\n")
            
            for stat in self.training_stats:
                f.write(f"Episode {stat['episode']}:\n")
                f.write(f"  Level: {stat['level']}\n")
                f.write(f"  Reward: {stat['reward']:.2f}\n")
                f.write(f"  Steps: {stat['steps']}\n")
                f.write(f"  Epsilon: {stat['epsilon']:.3f}\n")
                f.write("---------------\n")

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.store(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step"""
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
        goal = next(iter(self.level.goal_list))
        distance = ((goal.rect.centerx - self.agent.rect.centerx) ** 2 +
                    (goal.rect.centery - self.agent.rect.centery) ** 2) ** 0.5
        return distance
    
    def get_nearest_block_info(self):
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
        """Execute one time step within the environment"""
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
        """Reset the environment for a new episode"""
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
        if not self.render_enabled:
            raise ValueError("Cannot run game with visualization disabled")
        while self.running:
            self.events()
            self.update()
            self.draw()
            self.clock.tick(60)  # Limit to 60 FPS
        self.quit()

    def update(self):
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
        print("Restarting level...")
        # Reset agent position and other necessary states
        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0

    def level_completed(self):
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
        pygame.quit()
        sys.exit()

    def update_camera(self):
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

    args = parser.parse_args()

    game = Game(
        manual_control=args.m,
        level_number=args.l,
        training_mode=args.t,
        render_enabled=args.r if args.t else True,
        load_model=args.lm
    )
    
    if args.t:
        game.train_ai()
    else:
        game.run()