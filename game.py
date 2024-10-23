# game.py
import argparse
import pygame
import sys
import numpy as np
import torch
from collections import deque
import random
import os

from agent.agent import Agent
from world.levels.level1 import Level1
from world.levels.level2 import Level2
from world.levels.level3 import Level3
from world.dqn.dqn_agent import DQNAgent

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
BG_COLOR = (128, 128, 128)  # Gray color

class Game:
    def __init__(self, manual_control=False, level_number=1, training_mode=False, render_enabled=True):
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

        # Create the agent
        self.agent = Agent(50, SCREEN_HEIGHT - 80, screen_height=SCREEN_HEIGHT)
        self.load_level()
        
        # Sprite groups
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.level.get_all_sprites())
        self.all_sprites_list.add(self.agent)

        self.block_list = self.level.get_blocks()

        # Camera offset (initially zero)
        self.camera_x = 0
        self.camera_y = 0
        # Training stats
        self.consecutive_successes = {1: 0, 2: 0, 3: 0}  # Track successes per level
        self.training_stats = []  # Store episode results
        # Training-specific variables
        self.current_reward = 0
        self.episode_steps = 0
        self.max_steps = 1000  # Maximum steps per episode
        self.last_x_position = 50  # Track progress
        if training_mode:
            # Initialize DQN components
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dim = 8  # Size of our state vector
            action_dim = 4  # Number of possible actions
            self.dqn_agent = DQNAgent(state_dim, action_dim, self.device)
            self.replay_buffer = deque(maxlen=10000)
            self.batch_size = 64
            self.training_steps = 0
            self.target_update_freq = 10

    def train_ai(self, num_episodes=1000, successes_required=3):
        """Main training loop for AI"""
        for episode in range(num_episodes):
            self.restart_level()
            episode_reward = 0
            steps = 0
            current_level = self.level_number
            
            while self.running and steps < self.max_steps:
                state = self.get_state()
                action = self.dqn_agent.choose_action(state)
                
                # Execute action and get reward
                self.handle_ai_action(action)
                self.update()
                
                next_state = self.get_state()
                reward = self.get_reward()
                done = not self.running
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Train the network
                if len(self.replay_buffer) >= self.batch_size:
                    self.train_step()
                
                episode_reward += reward
                steps += 1
                
                if self.render_enabled:
                    self.draw()
                
                if done:
                    break
            
            # Record episode statistics
            episode_data = {
                'episode': episode + 1,
                'level': current_level,
                'reward': episode_reward,
                'steps': steps,
                'epsilon': self.dqn_agent.epsilon,
                'consecutive_successes': self.consecutive_successes[current_level]
            }
            self.training_stats.append(episode_data)
            
            # Print and save progress
            print(f"Episode {episode + 1}: Level = {current_level}, Reward = {episode_reward:.2f}, Steps = {steps}")
            print(f"Consecutive successes: {self.consecutive_successes}")
            
            # Save model periodically
            if episode % 100 == 0:
                self.dqn_agent.save(f"ai_model_ep{episode}.pth")
                self.save_training_stats()
            
            # Decay exploration rate
            self.dqn_agent.decay_epsilon()

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
                f.write(f"  Consecutive Successes: {stat['consecutive_successes']}\n")
                f.write("---------------\n")

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample random batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

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

    def get_state(self):
        """Get the current state for AI training"""
        # Find nearest trap
        nearest_trap_dist = float('inf')
        for trap in self.level.trap_list:
            dist = ((trap.rect.x - self.agent.rect.x) ** 2 + 
                   (trap.rect.y - self.agent.rect.y) ** 2) ** 0.5
            nearest_trap_dist = min(nearest_trap_dist, dist)

        # Find goal distance
        goal = next(iter(self.level.goal_list))
        goal_dist = ((goal.rect.x - self.agent.rect.x) ** 2 + 
                    (goal.rect.y - self.agent.rect.y) ** 2) ** 0.5

        # Check if on ground
        self.agent.rect.y += 2
        block_hit_list = pygame.sprite.spritecollide(self.agent, self.block_list, False)
        self.agent.rect.y -= 2
        on_ground = 1.0 if len(block_hit_list) > 0 else 0.0

        return np.array([
            self.agent.rect.x / self.level.width,
            self.agent.rect.y / self.level.height,
            self.agent.change_x / 10.0,
            self.agent.change_y / 10.0,
            min(1.0, nearest_trap_dist / 500.0),
            min(1.0, goal_dist / 1000.0),
            on_ground,
            float(self.level_number) / self.max_levels
        ], dtype=np.float32)

    def get_reward(self):
        """Calculate reward for the current state"""
        reward = 0
        
        # Reward for moving right (progress)
        progress = self.agent.rect.x - self.last_x_position
        reward += progress * 0.1
        self.last_x_position = self.agent.rect.x

        # Penalty for death
        if self.agent.rect.y > self.level.height:
            reward -= 100
            self.running = False

        # Check for trap collision
        trap_hit_list = pygame.sprite.spritecollide(self.agent, self.level.trap_list, False)
        if trap_hit_list:
            reward -= 100
            self.running = False

        # Reward for reaching goal
        goal_hit_list = pygame.sprite.spritecollide(self.agent, self.level.goal_list, False)
        if goal_hit_list:
            reward += 500
            self.level_completed()

        return reward

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

    def handle_ai_action(self, action):
        """Handle AI actions"""
        if action == 0:  # LEFT
            self.agent.go_left()
        elif action == 1:  # RIGHT
            self.agent.go_right()
        elif action == 2:  # JUMP
            self.agent.jump(self.block_list)
        elif action == 3:  # NOTHING
            self.agent.stop()

    def reset_episode(self):
        """Reset the environment for a new episode"""
        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0
        self.running = True
        self.episode_steps = 0
        self.last_x_position = 50
        self.current_reward = 0
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
        if self.agent.rect.y > self.level.height:
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
        print("Agent hit a trap! Restarting level...")
        # Reset agent position and other necessary states
        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0

    def level_completed(self):
        print(f"Level {self.level_number} Completed!")
        self.consecutive_successes[self.level_number] += 1
        
        if self.check_level_mastery():
            if self.level_number < self.max_levels:
                # Move to next level
                self.level_number += 1
                self.consecutive_successes[self.level_number] = 0  # Reset counter for new level
                print(f"Loading level {self.level_number}...")
                self.load_level()
            else:
                print("Congratulations! You've completed all levels!")
                self.running = False
        else:
            # Reset to retry current level
            print(f"Retrying level {self.level_number}. Consecutive successes: {self.consecutive_successes[self.level_number]}")
            self.restart_level()

    def draw(self):
        if not self.render_enabled:
            return
            
        self.screen.fill(BG_COLOR)
        for entity in self.all_sprites_list:
            self.screen.blit(entity.image, (entity.rect.x + self.camera_x, entity.rect.y + self.camera_y))
        pygame.display.flip()

    def check_level_mastery(self, successes_required=3):
        """Check if current level is mastered"""
        if self.consecutive_successes[self.level_number] >= successes_required:
            return True
        return False
    
    def quit(self):
        pygame.quit()
        sys.exit()

    def get_agent_inputs(self):
        # Placeholder for actual sensory inputs
        # You need to define how the agent perceives the environment
        # For example, distances to the nearest obstacles
        inputs = [0, 0, 0]  # Example inputs
        return inputs

    def handle_agent_action(self, action):
        if action == "left":
            self.agent.go_left()
        elif action == "right":
            self.agent.go_right()
        elif action == "jump":
            self.agent.jump(self.block_list)
        elif action == "stop":
            self.agent.stop()

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
                self.agent.jump(self.block_list)
        else:
            # AI control
            inputs = self.get_agent_inputs()
            action = self.agent.get_action(inputs)
            self.handle_agent_action(action)


    def load_level(self):
        # You can expand this dictionary as you add more levels
        level_classes = {
            1: Level1,
            2: Level2,
            3: Level3,
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
    parser.add_argument('--s', type=int, default=3, help='Number of consecutive successes required to advance.')
    args = parser.parse_args()

    game = Game(
        manual_control=args.m,
        level_number=args.l,
        training_mode=args.t,
        render_enabled=args.r if args.t else True
    )
    
    if args.t:
        game.train_ai(successes_required=args.s)
    else:
        game.run()