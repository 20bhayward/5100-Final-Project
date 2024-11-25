# trainer.py
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
        self.consecutive_successes = 0
        
        # State tracking
        self.last_state = None
        self.last_jump_time = 0
        self.jump_cooldown = 300  # milliseconds
        self.last_distance_to_goal = None
        
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
            self.dqn_agent.load(load_model)
            if not self.training_mode:
                self.dqn_agent.epsilon = 0.0

    def get_state(self):
        """Convert camera-based precepts to a state vector for the DQN."""
        visible_state = self.precepts.get_visible_state()
        
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
        distance_to_platform_edge = 1.0  # Initialize to max normalized distance
        
        if current_platform:
            current_rect = current_platform['rect']
            next_rect = next_platform['rect'] if next_platform else current_rect
            next_platform_dist = (next_rect.left - current_rect.right) / self.precepts.max_gap_width
            next_platform_height_diff = (current_rect.y - next_rect.y) / 100.0  # Normalize height difference
            
            # Calculate distance to the edge of the current platform
            distance_to_edge = current_rect.right - self.agent.rect.right
            distance_to_platform_edge = distance_to_edge / self.precepts.screen_width  # Normalize

        
        # Gap information
        gaps = platform_info['gaps']
        nearest_gap_dist = 1.0  # Normalized distance to nearest gap
        nearest_gap_width = 0.0  # Normalized gap width
        gap_jumpable = 0.0
        
        if gaps:
            nearest_gap = min(gaps, key=lambda g: abs(g['start_x'] - self.agent.rect.x))
            nearest_gap_dist = abs(nearest_gap['start_x'] - self.agent.rect.x) / self.precepts.screen_width
            nearest_gap_width = nearest_gap['width'] / self.precepts.max_gap_width
            gap_jumpable = float(nearest_gap['jumpable'])
        
        # Hazard information
        hazard_info = visible_state['hazards']
        nearest_hazard_dist = 1.0
        hazard_requires_jump = 0.0
        
        if hazard_info['immediate_threats']:
            nearest_threat = min(hazard_info['immediate_threats'], 
                               key=lambda t: t['distance'])
            nearest_hazard_dist = nearest_threat['distance'] / 100.0  # Normalize to [0,1]
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
        return 22  # Number of features in get_state()
    
    def get_reward(self):
        """Calculate reward based on the agent's actions and state."""
        state = self.get_state()
        
        # State indices based on get_state() in Trainer
        ON_PLATFORM = 0
        PLATFORM_EDGE_DIST = 5
        GAP_JUMPABLE = 8
        HAZARD_REQUIRES_JUMP = 10
        JUMP_SUCCESS_PROB = 12
        IS_JUMPING = 18
        
        reward = 0
        
        # Base movement reward - encourage moving right
        if self.agent.change_x > 0:
            reward += 1
        
        # Jumping rewards/penalties
        if state[IS_JUMPING]:  # Agent is jumping
            if state[GAP_JUMPABLE] > 0 or state[HAZARD_REQUIRES_JUMP] > 0:
                # Reward jumping when appropriate
                reward += 5
            else:
                # Penalize unnecessary jumps
                reward -= 2
        
        # Death penalties
        if self.agent.rect.y > self.level.height:  # Fell into pit
            reward -= 500
            self.pygame_manager.running = False
            return reward
                
        # Trap collision
        if pygame.sprite.spritecollide(self.agent, self.level.trap_list, False):
            reward -= 500
            self.pygame_manager.running = False
            return reward
        
        # Goal completion reward
        if pygame.sprite.spritecollide(self.agent, self.level.goal_list, False):
            reward += 400
            self.pygame_manager.running = False
            return reward
        
        # Small time penalty to encourage efficient completion
        reward -= 0.01
        
        return reward

    # def get_reward(self):
    #     """Calculate reward based on the agent's actions and state."""
    #     visible_state = self.precepts.get_visible_state()
    #     reward = 0

    #     # Platform and movement rewards
    #     if visible_state['movement']['on_ground']:
    #         if visible_state['movement']['velocity_x'] > 0:
    #             reward += 3  # Basic reward for moving forward
    #         platform_info = visible_state['platforms']
    #         if platform_info['current_platform']:
    #             reward += 1  # Small reward for being on a platform

    #     # Penalty for falling off the platform without jumping
    #     if not visible_state['movement']['on_ground'] and not visible_state['movement']['is_jumping']:
    #         reward -= 10  # Increased penalty for falling

    #     # Jump rewards
    #     jumps = visible_state['jumps']
    #     if jumps and visible_state['movement']['is_jumping']:
    #         best_jump = max(jumps, key=lambda j: j['success_prob'])
    #         if best_jump['success_prob'] > 0.8:
    #             reward += 10  # Increased reward for making a smart jump

    #     # Gap navigation rewards
    #     if visible_state['platforms']['gaps']:
    #         nearest_gap = min(visible_state['platforms']['gaps'],
    #                         key=lambda g: abs(g['start_x'] - self.agent.rect.x))
    #         if nearest_gap['jumpable']:
    #             if self.agent.rect.x > nearest_gap['start_x'] + nearest_gap['width']:
    #                 reward += 50  # Increased reward for clearing a gap
        
    #     # Hazard avoidance rewards
    #     hazards = visible_state['hazards']
    #     if hazards['immediate_threats']:
    #         for threat in hazards['immediate_threats']:
    #             if threat['requires_jump'] and not visible_state['movement']['is_jumping']:
    #                 reward -= 10  # Penalty for not jumping over a hazard
    #             elif threat['distance'] < 50:
    #                 reward -= 5  # Penalty for being too close to a hazard
        
    #      # Death penalties
    #     if self.agent.rect.y > self.level.height:
    #         reward -= 400
    #         print(f"Episode ended: Agent fell after {self.current_run_steps} steps")
    #         self.pygame_manager.running = False
    #         return reward
            
    #     # Trap collision
    #     trap_hit_list = pygame.sprite.spritecollide(self.agent, self.level.trap_list, False)
    #     if trap_hit_list:
    #         reward -= 400
    #         print(f"Episode ended: Agent hit trap after {self.current_run_steps} steps")
    #         self.pygame_manager.running = False
    #         return reward
        
    #     # Goal completion rewards
    #     goal_hit_list = pygame.sprite.spritecollide(self.agent, self.level.goal_list, False)
    #     if goal_hit_list:
    #         completion_time = self.current_run_steps
            
    #         # Base completion reward
    #         reward += 500
            
    #         # Time efficiency bonus
    #         time_bonus = max(0, 1000 - completion_time)
    #         reward += time_bonus * 0.5
            
    #         # New best time bonus
    #         if completion_time < self.best_completion_time:
    #             self.best_completion_time = completion_time
    #             reward += 250
            
    #         print(f"\nGoal reached! Steps: {completion_time} | Best: {self.best_completion_time}")
    #         self.pygame_manager.running = False
    #         return reward
        
    #     return reward
        
    def train_step(self):
        """Perform one step of training using the replay buffer."""
        if self.replay_buffer.size() < self.batch_size:
            return
            
        try:
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()
            
            # Print debug info about sampled data
            # print("\nTrainer Debug Info:")
            # print(f"Rewards stats - min: {rewards.min():.3f}, max: {rewards.max():.3f}, mean: {rewards.mean():.3f}")
            
            # Clip infinite values in states and next_states
            states = np.clip(states, -10.0, 10.0)
            next_states = np.clip(next_states, -10.0, 10.0)
            
            # print(f"States stats after clipping - min: {states.min():.3f}, max: {states.max():.3f}, mean: {states.mean():.3f}")
            # print(f"Number of terminal states: {sum(dones)}")
            
            # Train the network
            loss = self.dqn_agent.train(states, actions, rewards, next_states, dones)
            
            if loss is not None:
                self.training_steps += 1
                if self.training_steps % self.target_update_freq == 0:
                    print(f"\nUpdating target network at step {self.training_steps}")
                    self.dqn_agent.update_target_network()
            
            return loss
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if not self.pygame_manager.running:
            return self.get_state(), 0, True, {}
            
        # Execute the action
        self.handle_ai_action(action)
        
        # Update game state
        if self.pygame_manager.level:
            self.pygame_manager.update(self.pygame_manager.level, self.agent)
        
        # Get new state and reward
        next_state = self.get_state()
        reward = self.get_reward()
        
        # Update step counters
        self.episode_steps += 1
        self.current_run_steps += 1
        
        # Check if episode is done
        done = not self.pygame_manager.running or self.episode_steps >= 2000
        
        return next_state, reward, done, {}
    
    def train_ai(self, num_episodes):
        """
        Main training loop for the AI agent.
        
        Args:
            num_episodes (int): Number of episodes to train for
        """
        episode = 0
        self.training_active = True
        
        # Create save directory
        save_dir = f'trainer/trained_agents/level{self.level_number}'
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training on level {self.level_number}")
        print("Controls:")
        print("- Press 'q' to quit training")
        print("- Press 's' to save current progress")
        print(f"Will train for {num_episodes} episodes unless stopped...")
        
        while self.training_active and episode < num_episodes:
            episode += 1
            state = self.reset_episode()
            episode_reward = 0
            loss = 0
            
            # Reset moving platforms at the start of each episode
            for sprite in self.pygame_manager.get_all_sprites():
                if hasattr(sprite, 'is_moving') and sprite.is_moving:
                    sprite.rect.x = sprite.start_pos
                    sprite.speed = abs(sprite.speed)  # Reset direction
            
            while self.pygame_manager.is_running() and self.current_run_steps < 2000:
                # Handle Pygame events
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
            
            # Print progress
            if episode % 1 == 0:
                self.save_training_stats()
                print(f"\nEpisode {episode}/{num_episodes}:")
                print(f"  Steps: {self.current_run_steps}")
                print(f"  Distance: {self._get_distance_to_goal()}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Loss: {loss:.4f}" if loss else "  Loss: N/A")
                print(f"  Current ε: {self.dqn_agent.epsilon:.3f}")
                if self.best_completion_time != float('inf'):
                    print(f"  Best Completion Time: {self.best_completion_time}")
                print("------------------------")
            
            # Save checkpoint
            if episode % 100 == 0:
                checkpoint_path = f"{save_dir}/checkpoint_ep{episode}.pth"
                self.dqn_agent.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Decay exploration rate
            self.dqn_agent.decay_epsilon()
        
        # Final save
        print("\nTraining finished!")
        print(f"Completed {episode} episodes")
        self.save_training_stats()
        final_model_path = f"{save_dir}/final_model.pth"
        self.dqn_agent.save(final_model_path)
        print(f"Saved final model to {final_model_path}")

    def handle_ai_action(self, action_index):
        action = MOVEMENT_ACTIONS[action_index]
        
        visible_state = self.precepts.get_visible_state()
        current_platform = visible_state['platforms']['current_platform']
        next_platform = visible_state['platforms']['next_platform']
        movement = visible_state['movement']

        if action['move'] == 'right':
            self.agent.go_right()
        else:
            self.agent.stop()

        current_time = pygame.time.get_ticks()
        if action['jump'] and current_time - self.last_jump_time >= self.jump_cooldown:
            if movement['can_jump'] and current_platform and next_platform:
                # Get platform properties
                gap_width = next_platform['rect'].left - current_platform['rect'].right
                distance_to_edge = current_platform['rect'].right - self.agent.rect.right
                
                # Adjust for moving platforms
                if next_platform['is_moving']:
                    # Calculate time to reach platform
                    time_to_reach = gap_width / self.agent.max_speed_x
                    if next_platform['direction'] == 'horizontal':
                        # Predict platform position
                        platform_movement = next_platform['speed'] * time_to_reach
                        if platform_movement > 0:
                            gap_width += platform_movement
                        else:
                            gap_width -= platform_movement
                
                # Jump when close enough to edge and gap is jumpable
                if gap_width <= self.precepts.max_gap_width and distance_to_edge <= 15:
                    self.agent.jump()
                    self.last_jump_time = current_time
            
    def reset_episode(self):
        """
        Reset the environment and agent for a new episode.
        
        Returns:
            Initial state after reset
        """
        # Reset agent position and state
        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0
        
        # Reset moving platforms to their initial positions
        for sprite in self.pygame_manager.get_all_sprites():
            if hasattr(sprite, 'is_moving') and sprite.is_moving:
                sprite.rect.x = sprite.start_pos
                sprite.speed = abs(sprite.speed)  # Reset to initial direction
        
        self.pygame_manager.running = True
        
        # Reset episode-specific variables
        self.current_run_steps = 0
        self.episode_steps = 0
        self.last_jump_time = 0
        self.stuck_counter = 0
        self.last_progress_position = 50
        
        # Get initial state
        initial_state = self.get_state()
        visible_state = self.precepts.get_visible_state()
        self.last_distance_to_goal = self._get_distance_to_goal()
        
        # Reset success tracking on death
        if not self.pygame_manager.running:
            self.consecutive_successes = 0
            
        return initial_state

    def _get_distance_to_goal(self):
        """Calculate the horizontal distance to the goal."""
        goal = next(iter(self.level.goal_list))
        distance = abs(goal.rect.centerx - self.agent.rect.centerx)
        return distance


    def save_training_stats(self):
        """Save training statistics to files."""
        if not self.training_stats:
            print("Warning: No training stats to save!")
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