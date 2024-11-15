import torch
from core.config import ACTION_DIM, SCREEN_HEIGHT, MOVEMENT_ACTIONS
from dqn.dqn_agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
import pygame
import numpy as np
import os

class Trainer:
    def __init__(self, load_model=None, training_mode=True, pygame_manager=None, render_enabled=True, level=None,
                 level_number=1, agent=None):
        self.level_number = level_number
        self.agent = agent
        self.level = level
        self.training_active = False
        self.training_mode = training_mode
        self.training_stats = []
        self.render_enabled = render_enabled
        self.pygame_manager = pygame_manager

        # Initialize DQN components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = 10  # Updated state dimension
        action_dim = ACTION_DIM
        self.dqn_agent = DQNAgent(state_dim, action_dim, self.device)

        # Training-specific variables
        self.current_reward = 0
        self.episode_steps = 0
        self.max_steps = 5000  # Maximum steps per episode
        self.last_x_position = 50
        self.successful_completion_times = []  # Track only successful runs
        self.best_completion_time = float('inf')
        self.current_run_steps = 0
        self.last_action_index = 0
        self.last_jump_time = 0
        self.jump_cooldown = 400  # milliseconds

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=50000, batch_size=64)
        self.batch_size = 64
        self.training_steps = 0
        self.target_update_freq = 1000  # Adjusted target network update frequency

        if load_model:
            self.dqn_agent.load(load_model)
            if not self.training_mode:
                self.dqn_agent.epsilon = 0.0  # No exploration during inference

        # Calculate initial goal distance
        self.last_goal_distance = self.pygame_manager.game.precepts.get_goal_distance()

    def train_ai(self, num_episodes=1000):
        """
        Main training loop with explicit episode limit and better controls.
        """
        episode = 0
        self.training_active = True

        save_dir = f'trainer/trained_agents/level{self.level_number}'
        os.makedirs(save_dir, exist_ok=True)

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

            while self.pygame_manager.is_running() and self.current_run_steps < self.max_steps:
                for event in self.pygame_manager.event_handler():
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

                # Execute the action and get the next state
                next_state, reward, done, _ = self.step(action)

                self.store_experience(state, action, reward, next_state, done)

                if self.replay_buffer.size() >= self.batch_size:
                    self.train_step()

                episode_reward += reward
                self.current_run_steps += 1
                self.episode_steps += 1

                if self.render_enabled:
                    self.pygame_manager.draw(self.pygame_manager.get_all_sprites(), None)

                if done:
                    break

            episode_data = {
                'episode': episode,
                'level': self.level_number,
                'reward': float(episode_reward),
                'steps': self.current_run_steps,
                'epsilon': float(self.dqn_agent.epsilon)
            }
            self.training_stats.append(episode_data)

            if episode % 10 == 0:
                self.save_training_stats()
                print(f"\nEpisode {episode}/{num_episodes}:")
                print(f"  Steps: {self.current_run_steps}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Current ε: {self.dqn_agent.epsilon:.3f}")
                if self.best_completion_time != float('inf'):
                    print(f"  Best Completion Time: {self.best_completion_time}")
                print("------------------------")

            if episode % 100 == 0:
                checkpoint_path = f"{save_dir}/checkpoint_ep{episode}.pth"
                self.dqn_agent.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            self.dqn_agent.decay_epsilon()

        print("\nTraining finished!")
        print(f"Completed {episode} episodes")
        self.save_training_stats()
        final_model_path = f"{save_dir}/final_model.pth"
        self.dqn_agent.save(final_model_path)
        print(f"Saved final model to {final_model_path}")

    def save_training_stats(self):
        if not self.training_stats:
            print("Warning: No training stats to save!")
            return

        # Create directories if they don't exist
        save_dir = f'trainer/trained_agents/level{self.level_number}'
        try:
            os.makedirs(save_dir, exist_ok=True)

            # Save as TXT
            with open(f'{save_dir}/training_results.txt', 'w') as f:
                f.write("Training Results\n===============\n\n")

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
                f.write("Episode Details\n---------------\n")
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
            df.to_csv(f'{save_dir}/training_results.csv', index=False)

            print(f"Successfully saved {len(self.training_stats)} training episodes to {save_dir}")

        except Exception as e:
            print(f"Error saving training stats: {e}")
            import traceback
            traceback.print_exc()

        except Exception as e:
            print(f"Error saving training stats: {e}")
            import traceback
            traceback.print_exc()

    def store_experience(self, state, action, reward, next_state, done):
        """Store the experience tuple in the replay buffer."""
        self.replay_buffer.store(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step using a batch from the replay buffer."""
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        loss = self.dqn_agent.train(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

        self.training_steps += 1

        if self.training_steps % self.target_update_freq == 0:
            self.dqn_agent.update_target_network()

    def step(self, action):
        """Execute one step in the environment."""
        if not self.pygame_manager.running:
            return self.get_state(), 0, True, {}

        # Convert action to game commands
        self.handle_ai_action(action)

        # Update game state
        if self.pygame_manager.level:
            self.pygame_manager.update(self.pygame_manager.level, self.agent)

        # Get reward and next state
        reward = self.get_reward()
        next_state = self.get_state()

        # Check if episode is done
        self.episode_steps += 1
        done = not self.pygame_manager.is_running() or self.episode_steps >= self.max_steps

        return next_state, reward, done, {}

    def get_state(self):
        """Get the current state from the environment."""
        # Get agent state information
        agent_x = self.agent.rect.x / self.level.width
        agent_y = self.agent.rect.y / self.level.height
        agent_vel_x = self.agent.change_x / self.agent.max_speed_x
        agent_vel_y = self.agent.change_y / self.agent.terminal_velocity

        # Get environmental information from precepts
        goal_distance = self.pygame_manager.game.precepts.get_goal_distance() / (self.level.width + self.level.height)
        on_ground = 1.0 if self.agent.on_ground else 0.0
        nearest_block_dist, obstacle_in_front = self.pygame_manager.game.precepts.get_nearest_block_info()
        trap_ahead = self.pygame_manager.game.precepts.is_trap_ahead()
        trap_distance = self.pygame_manager.game.precepts.get_nearest_trap_distance() / 500.0

        return np.array([
            agent_x,
            agent_y,
            agent_vel_x,
            agent_vel_y,
            goal_distance,
            on_ground,
            obstacle_in_front,
            nearest_block_dist / 500.0,
            trap_ahead,
            trap_distance
        ], dtype=np.float32)

    def get_reward(self):
        """Calculate the reward for the current state."""
        reward = 0

        # Distance-based reward
        current_goal_distance = self.pygame_manager.game.precepts.get_goal_distance()
        distance_progress = self.last_goal_distance - current_goal_distance
        reward += distance_progress * 0.1
        self.last_goal_distance = current_goal_distance

        # Penalties
        if distance_progress < 0:
            reward += distance_progress * 0.5  # Penalty for moving backward

        if self.agent.change_x == 0:
            reward -= 0.1  # Penalty for standing still

        # Trap penalties
        trap_ahead = self.pygame_manager.game.precepts.is_trap_ahead()
        trap_distance = self.pygame_manager.game.precepts.get_nearest_trap_distance()
        if trap_ahead:
            reward -= (1 - (trap_distance / 500.0)) * 0.1

        # Death penalties
        if self.agent.rect.y > self.level.height * 2:  # Fell out of bounds
            reward -= 100
            self.pygame_manager.running = False
            print(f"Failed: Fell out of bounds after {self.current_run_steps} steps")
            return reward

        # Check collisions
        trap_hit_list = pygame.sprite.spritecollide(self.agent, self.level.trap_list, False)
        if trap_hit_list:
            reward -= 200
            self.pygame_manager.running = False
            print(f"Failed: Hit trap after {self.current_run_steps} steps")
            return reward

        # Goal completion reward
        goal_hit_list = pygame.sprite.spritecollide(self.agent, self.level.goal_list, False)
        if goal_hit_list:
            if self.current_run_steps < self.best_completion_time:
                self.best_completion_time = self.current_run_steps
                print(f"\nNew best completion time: {self.best_completion_time} steps!")
                save_dir = f'trainer/trained_agents/level{self.level_number}'
                best_model_path = f"{save_dir}/best_model.pth"
                self.dqn_agent.save(best_model_path)
                print(f"Saved new best model to {best_model_path}")

            time_bonus = max(0, 1000 - self.current_run_steps)
            reward += 500 + time_bonus
            self.successful_completion_times.append(self.current_run_steps)
            print(f"Level completed in {self.current_run_steps} steps!")
            if self.successful_completion_times:
                avg_time = sum(self.successful_completion_times[-10:]) / len(self.successful_completion_times[-10:])
                print(f"Average completion time: {avg_time:.1f} steps")
            self.pygame_manager.running = False

        return reward

    def handle_ai_action(self, action_index):
        """Convert action index to game commands and execute them."""
        command = MOVEMENT_ACTIONS[action_index]
        self.last_action_index = action_index

        # Handle movement
        if command == 'left':
            self.agent.go_left()
        elif command == 'right':
            self.agent.go_right()
        else:
            self.agent.stop()

        # Handle jumping
        if self.pygame_manager.game.precepts.is_jump_necessary():
            current_time = pygame.time.get_ticks()
            if current_time - self.last_jump_time >= self.jump_cooldown:
                self.agent.jump()
                self.last_jump_time = current_time

    def reset_episode(self):
        """Reset the environment for a new episode."""
        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0
        self.pygame_manager.running = True
        self.last_x_position = 50
        self.current_reward = 0
        self.current_run_steps = 0
        self.last_jump_time = 0
        self.last_goal_distance = self.pygame_manager.game.precepts.get_goal_distance()
        return self.get_state()
