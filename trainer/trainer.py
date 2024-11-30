import torch
import numpy as np
import pygame
import os
import json
import pandas as pd
from core.config import ACTION_DIM, SCREEN_HEIGHT, MOVEMENT_ACTIONS
from dqn.dqn_agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from agent.camera_precepts import CameraBasedPrecepts

class Trainer:
    def __init__(self, load_model=None, training_mode=True, pygame_manager=None,
                 render_enabled=True, level=None, level_number=1, agent=None):
        """Initialize the trainer with camera-based precepts."""
        self.level_number = level_number
        self.agent = agent
        self.level = level
        self.training_mode = training_mode
        self.render_enabled = render_enabled
        self.pygame_manager = pygame_manager

        self.jump_start_position = None
        self.crossed_gaps = set()  # Track which gaps have been crossed

        # Initialize new camera-based precepts
        self.precepts = CameraBasedPrecepts(agent, level, pygame_manager)

        # Training state
        self.training_active = False
        self.training_stats = []
        self.current_reward = 0
        self.episode_steps = 0
        self.current_run_steps = 0

        # Performance tracking
        self.successful_completion_times = []
        self.best_completion_time = float('inf')

        # Progress tracking
        self.last_x_position = None
        self.steps_without_progress = 0
        self.max_steps_without_progress = 100  # Reduced for faster termination
        self.last_distance_to_goal = None
        self.last_significant_progress = 0
        self.last_jump_time = 0
        self.jump_cooldown = 200  # milliseconds, reduced for more frequent jumping

        # Initialize DQN components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = self._calculate_state_dim()
        self.dqn_agent = DQNAgent(state_dim, ACTION_DIM, self.device)

        # Enhanced replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=128)
        self.batch_size = 128
        self.training_steps = 0
        self.target_update_freq = 500

        # Load pretrained model if provided
        if load_model:
            try:
                print(f"\nLoading model from: {load_model}")
                if not os.path.exists(load_model):
                    raise FileNotFoundError(f"Model file not found: {load_model}")

                self.dqn_agent.load(load_model)
                print("Model loaded successfully!")

                if not training_mode:
                    self.dqn_agent.epsilon = 0.0
                    print("\nEvaluation mode: epsilon set to 0.0")

            except Exception as e:
                print(f"Error during model loading/testing: {str(e)}")
                raise


    def get_state(self):
        """Convert camera-based precepts to a state vector for the DQN."""
        visible_state = self.precepts.get_visible_state()

        # Ensure ground detection
        if self.agent.rect.y >= SCREEN_HEIGHT - 80:  # If at starting height
            visible_state['movement']['on_ground'] = True
            visible_state['movement']['can_jump'] = True

        # Force platform detection at start
        if self.agent.rect.x <= 100:
            platform_info = visible_state['platforms']
            if not platform_info['current_platform']:
                # Create synthetic platform info for starting position
                platform_info['current_platform'] = {
                    'rect': pygame.Rect(0, SCREEN_HEIGHT - 60, 200, 20),
                    'is_moving': False
                }

        # Platform information
        platform_info = visible_state['platforms']
        current_platform = platform_info['current_platform']
        next_platform = platform_info['next_platform']

        current_platform_moving = float(current_platform['is_moving']) if current_platform else 0.0
        next_platform_moving = float(next_platform['is_moving']) if next_platform else 0.0

        # Initialize platform features
        on_platform = float(current_platform is not None)
        next_platform_dist = 0.0
        next_platform_height_diff = 0.0
        distance_to_platform_edge = 1.0

        if current_platform:
            current_rect = current_platform['rect']
            next_rect = next_platform['rect'] if next_platform else current_rect
            next_platform_dist = (next_rect.left - current_rect.right) / self.precepts.max_gap_width
            next_platform_height_diff = (current_rect.y - next_rect.y) / 100.0
            distance_to_edge = current_rect.right - self.agent.rect.right
            distance_to_platform_edge = distance_to_edge / self.precepts.screen_width

        # Gap information
        gaps = platform_info['gaps']
        nearest_gap_dist = 1.0
        nearest_gap_width = 0.0
        gap_jumpable = 0.0
        over_gap = False

        if gaps:
            nearest_gap = min(gaps, key=lambda g: abs(g['start_x'] - self.agent.rect.x))
            nearest_gap_dist = abs(nearest_gap['start_x'] - self.agent.rect.x) / self.precepts.screen_width
            nearest_gap_width = nearest_gap['width'] / self.precepts.max_gap_width
            gap_jumpable = float(nearest_gap['jumpable'])
            over_gap = nearest_gap['start_x'] <= self.agent.rect.centerx <= nearest_gap['start_x'] + nearest_gap['width']

        # Hazard information
        hazard_info = visible_state['hazards']
        nearest_hazard_dist = 1.0
        hazard_requires_jump = 0.0

        if hazard_info['immediate_threats']:
            nearest_threat = min(hazard_info['immediate_threats'],
                               key=lambda t: t['distance'])
            nearest_hazard_dist = nearest_threat['distance'] / 100.0
            hazard_requires_jump = float(nearest_threat['requires_jump'])

        # Jump opportunities
        jumps = visible_state['jumps']
        best_jump_prob = 0.0
        optimal_jump_dist = 1.0

        if jumps:
            best_jump = max(jumps, key=lambda j: j['success_prob'])
            best_jump_prob = best_jump['success_prob']
            optimal_jump_dist = abs(best_jump['start_x'] - self.agent.rect.x) / self.precepts.screen_width

        # Movement state
        movement = visible_state['movement']

        # Combine all features into state vector
        state = np.array([
            # Platform features
            on_platform,
            next_platform_dist,
            next_platform_height_diff,
            current_platform_moving,
            next_platform_moving,
            distance_to_platform_edge,

            # Gap features
            nearest_gap_dist,
            nearest_gap_width,
            gap_jumpable,
            float(over_gap),

            # Hazard features
            nearest_hazard_dist,
            hazard_requires_jump,
            float(hazard_info['threat_count'] > 0),

            # Jump features
            best_jump_prob,
            optimal_jump_dist,
            float(len(jumps) > 0),

            # Movement features
            movement['velocity_x'],
            movement['velocity_y'],
            float(movement['on_ground']),
            float(movement['is_jumping']),
            float(movement['can_jump']),

            # Goal features
            float(visible_state['goal_visible']),
            visible_state['goal_direction']
        ], dtype=np.float32)

        # Safety checks
        state = np.clip(state, -1, 1)
        if np.isnan(state).any() or np.isinf(state).any():
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        return state

    def _calculate_state_dim(self):
        """Calculate the dimension of the state vector."""
        return 23  # Number of features in get_state()

    def handle_ai_action(self, action_index):
        """Enhanced action handler with forced movement."""
        action = MOVEMENT_ACTIONS[action_index]

        # Get current state
        visible_state = self.precepts.get_visible_state()
        current_x = self.agent.rect.x
        velocity_x = visible_state['movement']['velocity_x']

        # Execute chosen action
        if action['move'] == 'right':
            self.agent.go_right()
        elif action['move'] == 'nothing':
            self.agent.stop()

        # Handle jumping
        current_time = pygame.time.get_ticks()
        if action['jump'] and current_time - self.last_jump_time >= self.jump_cooldown:
            if visible_state['movement']['can_jump']:
                self.agent.jump()
                self.last_jump_time = current_time

    def get_reward(self):
        """Reward function with guaranteed return value."""
        visible_state = self.precepts.get_visible_state()
        reward = 0

        # Track position and progress
        current_x = self.agent.rect.x
        if self.last_x_position is None:
            self.last_x_position = current_x
            return 0  # Return 0 for first call when last_x_position is None

        # Track jump start for gap crossing detection
        if self.agent.is_jumping and self.jump_start_position is None:
            self.jump_start_position = current_x
        elif not self.agent.is_jumping and self.jump_start_position is not None:
            # Check for gap crossing upon landing
            platform_info = visible_state['platforms']
            if current_platform := platform_info['current_platform']:
                for gap in platform_info['gaps']:
                    gap_start = gap['start_x']
                    gap_end = gap_start + gap['width']

                    if (gap_start not in self.crossed_gaps and
                        self.jump_start_position < gap_start and
                        current_x > gap_end):
                        print(f"Successfully jumped gap! ({gap_start} to {gap_end})")
                        self.crossed_gaps.add(gap_start)
                        reward += 100.0

            self.jump_start_position = None

        # Penalty for unnecessary jumping
        if self.agent.is_jumping and not any(gap['jumpable'] for gap in visible_state['platforms']['gaps']):
            reward -= 1.0

        # Regular movement rewards
        x_progress = current_x - self.last_x_position
        if x_progress > 0:
            reward += x_progress * 0.5
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
            reward -= self.steps_without_progress * 0.5

        # Terminal states (all return rewards directly)
        if self.agent.rect.y > self.level.height:  # Fell into pit
            self.pygame_manager.running = False
            return -100

        if pygame.sprite.spritecollide(self.agent, self.level.trap_list, False):
            self.pygame_manager.running = False
            return -100

        if pygame.sprite.spritecollide(self.agent, self.level.goal_list, False):
            self.pygame_manager.game.level_completed()
            self.pygame_manager.running = False
            return 200

        if self.steps_without_progress >= self.max_steps_without_progress:
            self.pygame_manager.running = False
            return -50

        # Update position tracking and return final reward
        self.last_x_position = current_x
        return reward

    # In Trainer class
    def step(self, action):
        """Execute one step with proper action handling."""
        if not self.pygame_manager.running:
            return self.get_state(), 0, True, {}

        # Ensure action is within bounds
        action = max(0, min(action, len(MOVEMENT_ACTIONS) - 1))

        # Execute action
        self.handle_ai_action(action)

        # Update game state
        if self.pygame_manager.level:
            self.pygame_manager.update(self.pygame_manager.level, self.agent)

        # Get new state and reward
        next_state = self.get_state()
        reward = self.get_reward()

        # Update counters
        self.episode_steps += 1
        self.current_run_steps += 1

        # Check termination
        done = (not self.pygame_manager.running or
                self.episode_steps >= 1000 or
                self.steps_without_progress >= self.max_steps_without_progress)

        return next_state, reward, done, {}


    def train_ai(self, num_episodes):
        """Main training loop."""
        episode = 0
        self.training_active = True

        # Create save directory
        save_dir = f'trainer/trained_agents/level{self.level_number}'
        os.makedirs(save_dir, exist_ok=True)

        while self.training_active and episode < num_episodes:
            episode += 1
            state = self.reset_episode()
            episode_reward = 0
            loss = 0

            # Reset moving platforms
            for sprite in self.pygame_manager.get_all_sprites():
                if hasattr(sprite, 'is_moving') and sprite.is_moving:
                    sprite.rect.x = sprite.start_pos
                    sprite.speed = abs(sprite.speed)

            while self.pygame_manager.is_running() and self.current_run_steps < 1000:
                # Handle Pygame events
                for event in self.pygame_manager.event_handler():
                    if event.type == pygame.QUIT:
                        self.training_active = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.training_active = False
                            break
                        elif event.key == pygame.K_s:
                            self.save_training_stats()
                            self.dqn_agent.save(f"{save_dir}/manual_save_ep{episode}.pth")

                if not self.training_active:
                    break

                # Choose and perform action
                action = self.dqn_agent.choose_action(state)
                next_state, reward, done, _ = self.step(action)

                # Store experience
                self.replay_buffer.store(state, action, reward, next_state, done)

                # Train the network
                if self.replay_buffer.size() >= self.batch_size:
                    loss = self.train_step()

                state = next_state
                episode_reward += reward

                if self.render_enabled:
                    self.pygame_manager.draw(self.pygame_manager.get_all_sprites(), None)
                    self.pygame_manager.tick(60)  # Lock to 60 FPS

                if done:
                    break

            # Record episode statistics
            episode_data = {
                'episode': episode,
                'level': self.level_number,
                'reward': float(episode_reward),
                'steps': self.current_run_steps,
                'loss': float(loss) if loss else 0.0,
                'epsilon': float(self.dqn_agent.epsilon)
            }
            self.training_stats.append(episode_data)

            # Save checkpoint
            if episode % 1 == 0:
                self.save_training_stats()
                print(f"\nEpisode {episode}/{num_episodes}:")
                print(f"  Steps: {self.current_run_steps}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Loss: {loss:.4f}" if loss else "  Loss: N/A")
                print(f"  Current ε: {self.dqn_agent.epsilon:.3f}")
                if self.best_completion_time != float('inf'):
                    print(f"  Best Completion Time: {self.best_completion_time}")
                print("------------------------")

            if episode % 100 == 0:
                checkpoint_path = f"{save_dir}/checkpoint_ep{episode}.pth"
                self.dqn_agent.save(checkpoint_path)

            # Decay exploration rate
            self.dqn_agent.decay_epsilon()

        # Final save
        self.save_training_stats()
        final_model_path = f"{save_dir}/final_model.pth"
        self.dqn_agent.save(final_model_path)

    def train_step(self):
        """Perform one step of training using the replay buffer."""
        if self.replay_buffer.size() < self.batch_size:
            return

        try:
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()

            # Clip infinite values in states and next_states
            states = np.clip(states, -10.0, 10.0)
            next_states = np.clip(next_states, -10.0, 10.0)

            # Train the network
            loss = self.dqn_agent.train(states, actions, rewards, next_states, dones)

            if loss is not None:
                self.training_steps += 1
                if self.training_steps % self.target_update_freq == 0:
                    self.dqn_agent.update_target_network()

            return loss

        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return None

    def verify_loaded_weights(self):
        """Verify model weights are properly loaded."""
        for name, param in self.dqn_agent.q_network.named_parameters():
            print(f"\nLayer: {name}")
            print(f"- Mean: {param.mean().item():.6f}")
            print(f"- Std: {param.std().item():.6f}")
            print(f"- Min: {param.min().item():.6f}")
            print(f"- Max: {param.max().item():.6f}")

    def _verify_model_loaded(self):
        """Verify that the model was loaded correctly."""
        try:
            for name, param in self.dqn_agent.q_network.named_parameters():
                if torch.all(param == 0) or torch.isnan(param).any():
                    print(f"Warning: Layer {name} appears invalid!")
                    return False
                print(f"\nLayer: {name}")
                print(f"  Mean: {param.mean().item():.6f}")
                print(f"  Std: {param.std().item():.6f}")
                print(f"  Min: {param.min().item():.6f}")
                print(f"  Max: {param.max().item():.6f}")
            return True
        except Exception as e:
            print(f"Error during model verification: {str(e)}")
            return False

    def reset_episode(self):
        """Reset episode with cleaned up progress tracking."""
        # Reset agent position and state
        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0

        # Reset moving platforms
        for sprite in self.pygame_manager.get_all_sprites():
            if hasattr(sprite, 'is_moving') and sprite.is_moving:
                sprite.rect.x = sprite.start_pos
                sprite.speed = abs(sprite.speed)

        self.pygame_manager.running = True

        # Reset progress tracking
        self.current_run_steps = 0
        self.episode_steps = 0
        self.last_jump_time = 0
        self.steps_without_progress = 0
        self.last_x_position = None
        self.last_significant_progress = 0

        # Reset jump tracking
        self.jump_start_position = None
        self.crossed_gaps.clear()

        # Get initial state
        initial_state = self.get_state()

        return initial_state

    def save_training_stats(self):
        """Save training statistics to files."""
        if not self.training_stats:
            return

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
                    if 'loss' in stat:
                        f.write(f"  Loss: {stat['loss']:.4f}\n")
                    f.write("---------------\n")

            # Save as CSV
            df = pd.DataFrame(self.training_stats)
            df.to_csv(f'{save_dir}/training_results.csv', index=False)

            # Save as JSON
            with open(f'{save_dir}/training_results.json', 'w') as f:
                json.dump(self.training_stats, f, indent=2)

        except Exception as e:
            print(f"Error saving training stats: {e}")
            import traceback
            traceback.print_exc()
