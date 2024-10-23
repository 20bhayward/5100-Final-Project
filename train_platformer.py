# train_platformer.py

import torch
import numpy as np
from game import Game
from world.dqn_agent import DQNAgent
from world.replay_buffer import ReplayBuffer
from world.q_network import QNetwork
import os
import time

def train():
    # Training settings
    BATCH_SIZE = 64
    GAMMA = 0.99
    TARGET_UPDATE = 10
    MEMORY_SIZE = 10000
    LEARNING_RATE = 0.001
    EPISODES = 1000
    
    # Initialize game in training mode
    game = Game(manual_control=False, level_number=1, training_mode=True)
    
    # Initialize DQN
    state_size = 8  # Size of our state vector
    action_size = 4  # Number of possible actions (LEFT, RIGHT, JUMP, NOTHING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize agent and replay buffer
    agent = DQNAgent(state_size, action_size, device)
    replay_buffer = ReplayBuffer(MEMORY_SIZE, BATCH_SIZE)

    # Training stats
    best_reward = float('-inf')
    level_completion_count = {1: 0, 2: 0, 3: 0}
    episode_rewards = []
    
    # Create directory for saving models
    os.makedirs('trained_models', exist_ok=True)

    print("Starting training...")
    print(f"Device: {device}")
    
    for episode in range(EPISODES):
        state = game.reset_episode()
        episode_reward = 0
        steps = 0
        start_time = time.time()
        
        # Episode loop
        while True:
            # Select action
            action = agent.choose_action(state)
            
            # Execute action
            next_state, reward, done, _ = game.step(action)
            episode_reward += reward
            
            # Store transition
            replay_buffer.store(state, action, reward, next_state, done)
            
            # Train the network
            if replay_buffer.size() >= BATCH_SIZE:
                agent.train(replay_buffer, BATCH_SIZE)
            
            state = next_state
            steps += 1
            
            # Optional: render game (can be disabled for faster training)
            game.draw()
            
            if done:
                episode_time = time.time() - start_time
                episode_rewards.append(episode_reward)
                
                # Print episode stats
                print(f"Episode {episode + 1}")
                print(f"Steps: {steps}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Level: {game.level_number}")
                print(f"Time: {episode_time:.2f}s")
                print("------------------------")
                
                # Save best performing model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save(f"trained_models/best_model.pth")
                    print(f"New best reward: {best_reward:.2f}")
                
                # Save checkpoint every 100 episodes
                if episode % 100 == 0:
                    agent.save(f"trained_models/checkpoint_ep{episode}.pth")
                
                # Track level completions
                if game.level_number > 1:  # Level was completed
                    level_completion_count[game.level_number - 1] += 1
                    
                # Update target network periodically
                if episode % TARGET_UPDATE == 0:
                    agent.update_target_network()
                
                break
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Print training statistics every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"\nLast 10 episodes average reward: {avg_reward:.2f}")
            print(f"Level completion counts: {level_completion_count}")
            print(f"Current exploration rate: {agent.epsilon:.3f}")
            print("------------------------\n")

def test_trained_agent(model_path, num_episodes=5):
    """Test a trained model"""
    game = Game(manual_control=False, level_number=1, training_mode=False)
    
    state_size = 8
    action_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DQNAgent(state_size, action_size, device)
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during testing
    
    for episode in range(num_episodes):
        state = game.reset_episode()
        total_reward = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = game.step(action)
            total_reward += reward
            state = next_state
            
            game.draw()  # Render the game
            
            if done:
                print(f"Test Episode {episode + 1}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Level Reached: {game.level_number}")
                print("------------------------")
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, help='Path to model file for testing')
    args = parser.parse_args()
    
    if args.test:
        test_trained_agent(args.test)
    else:
        train()