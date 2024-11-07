#!/usr/bin/env python3
import pygame
import torch
from game import Game
from world.dqn.dqn_agent import DQNAgent
import sys

# Configuration variables - modify these as needed
MODEL_PATH = "trained_models/best_model_2.pth"  # Path to your .pth file
LEVEL = 2                      # Level to run (1-3)
NUM_EPISODES = 50            # Number of episodes to run
RENDER_ENABLED = True         # Set to False to disable visualization
SPEED = "normal"                # "normal" or "fast"

def run_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = DQNAgent(8, 4, device)
    try:
        agent.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    agent.epsilon = 0  # No exploration

    try:
        delay = 16 if SPEED == "normal" else 0
        pygame.init()

        for episode in range(NUM_EPISODES):
            print(f"\nStarting Episode {episode + 1}/{NUM_EPISODES}")

            # Reinitialize the game for each episode
            game = Game(
                manual_control=False,
                level_number=LEVEL,
                training_mode=False,
                render_enabled=RENDER_ENABLED
            )

            state = game.reset_episode()
            if state is None:
                print("Error: Failed to get initial state")
                break

            total_reward = 0
            steps = 0

            while game.running:  # Use game.running as the loop condition
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and
                        event.key == pygame.K_ESCAPE
                    ):
                        print("Received quit signal")
                        return

                # Run agent
                try:
                    action = agent.choose_action(state)
                    next_state, reward, done, info = game.step(action)

                    if next_state is None:
                        print("Error: Invalid state returned from step")
                        break

                    total_reward += reward
                    steps += 1

                    # Update display if enabled
                    if RENDER_ENABLED:
                        game.draw()
                        if delay:
                            pygame.time.wait(delay)

                    state = next_state

                    if done:
                        print(f"Episode {episode + 1} finished: Steps = {steps}, Reward = {total_reward:.2f}")
                        break

                except Exception as e:
                    print(f"Error during episode step: {e}")
                    break

            # Add a small delay between episodes
            pygame.time.wait(100)

            # Check for manual exit between episodes
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and
                    event.key == pygame.K_ESCAPE
                ):
                    print("Received quit signal between episodes")
                    return

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        pygame.quit()
        print("Game closed")

if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
