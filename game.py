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
from world.levels.level4 import Level4
from world.levels.level5 import Level5
from world.levels.level6 import Level6
from world.levels.level7 import Level7
from world.levels.level8 import Level8
from world.dqn.dqn_agent import DQNAgent

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
BG_COLOR = (255, 255, 255)  # Gray color

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

    def train_ai(self, num_episodes=1000):
        """Main training loop focusing on single level mastery"""
        episode = 0
        
        print(f"Starting training on level {self.level_number}")
        print("Training will continue until manually stopped...")
        
        while True:
            episode += 1
            self.reset_episode()
            episode_reward = 0
            
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
                
                if len(self.replay_buffer) >= self.batch_size:
                    self.train_step()
                
                episode_reward += reward
                self.current_run_steps += 1
                
                if self.render_enabled:
                    self.draw()
                
                if done:
                    break
            
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
        """Calculate reward with jump penalties"""
        reward = 0
        
        # Progress reward
        progress = self.agent.rect.x - self.last_x_position
        reward += progress * 0.1
        self.last_x_position = self.agent.rect.x
        
        # Penalty for jumping
        if self.agent.change_y < 0:  # Agent is moving upward (jumping)
            reward -= 0.6  # penalty for jumping
        
        # Small time penalty to encourage speed
        reward -= 0.01

        # Death penalties
        if self.agent.rect.y > self.level.height * 2:
            reward -= 100
            self.running = False
            print(f"Failed: Fell out of bounds after {self.current_run_steps} steps")
            return reward

        # Trap collision
        trap_hit_list = pygame.sprite.spritecollide(self.agent, self.level.trap_list, False)
        if trap_hit_list:
            reward -= 100
            self.running = False
            print(f"Failed: Hit trap after {self.current_run_steps} steps")
            return reward

        # Goal completion
        goal_hit_list = pygame.sprite.spritecollide(self.agent, self.level.goal_list, False)
        if goal_hit_list:
            # Record completion time
            if self.current_run_steps < self.best_completion_time:
                self.best_completion_time = self.current_run_steps
                print(f"\nNew best completion time: {self.best_completion_time} steps!")
                # Save best performing model
                self.dqn_agent.save(f"best_model_level{self.level_number}.pth")
            
            # Reward based on completion speed
            time_bonus = max(0, 1000 - self.current_run_steps)
            reward += 500 + time_bonus  # Base completion reward plus time bonus
            
            self.successful_completion_times.append(self.current_run_steps)
            print(f"Level completed in {self.current_run_steps} steps!")
            print(f"Average completion time: {sum(self.successful_completion_times[-10:]) / len(self.successful_completion_times[-10:]):.1f} steps")
            self.running = False
            
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
        """Handle AI actions with jump cooldown"""
        current_time = pygame.time.get_ticks()
        
        if action == 0:  # LEFT
            self.agent.go_left()
        elif action == 1:  # RIGHT
            self.agent.go_right()
        elif action == 2:  # JUMP
            # Only allow jumping if cooldown has passed
            if current_time - self.last_jump_time >= self.jump_cooldown:
                self.agent.jump(self.block_list)
                self.last_jump_time = current_time
        elif action == 3:  # NOTHING
            self.agent.stop()

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
        if self.agent.rect.y > self.level.height*2:
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
    args = parser.parse_args()

    game = Game(
        manual_control=args.m,
        level_number=args.l,
        training_mode=args.t,
        render_enabled=args.r if args.t else True
    )
    
    if args.t:
        game.train_ai()
    else:
        game.run()