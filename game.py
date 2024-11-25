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
from core.pygame_manager import PygameManager
from core.config import SCREEN_WIDTH, SCREEN_HEIGHT, ACTION_DIM, MOVEMENT_ACTIONS
from trainer.trainer import Trainer
from agent.agent_precepts_old import AgentPrecepts

class Game:
    def __init__(self, manual_control=False, level_number=1, training_mode=False, render_enabled=True, load_model=None):
        """
        Initialize the game environment.
        """
        self.agent = Agent(50, SCREEN_HEIGHT - 80, screen_height=SCREEN_HEIGHT)
        self.pygame_manager = PygameManager(render_enabled, self, agent=self.agent)
        self.screen = self.pygame_manager.get_screen()

        self.level_number = level_number
        self.max_levels = 3
        self.load_level()  # Load level first so it's available for other components

        self.manual_control = manual_control
        self.training_mode = training_mode
        self.render_enabled = render_enabled

        # Add level to PygameManager after initialization
        self.pygame_manager.level = self.level

        # Add sprites to PygameManager
        self.pygame_manager.add_sprites(self.level.get_all_sprites())
        self.pygame_manager.add_sprites(self.agent)
        self.pygame_manager.set_blocks(self.level.get_blocks())

        # Add jump cooldown tracking
        self.last_jump_time = 0
        self.jump_cooldown = 400  # milliseconds

        # Create AgentPrecepts before Trainer
        self.precepts = AgentPrecepts(self.agent, self.level, self.pygame_manager)

        # Initialize Trainer with all necessary components
        self.trainer = Trainer(
            load_model=load_model,
            training_mode=training_mode,
            pygame_manager=self.pygame_manager,
            render_enabled=render_enabled,
            level=self.level,
            level_number=level_number,
            agent=self.agent
        )

        # Add trainer reference to PygameManager
        self.pygame_manager.trainer = self.trainer

    def get_agent(self):
        return self.agent

    def run(self):
        """
        Runs the main game loop.
        """
        if not self.render_enabled:
            raise ValueError("Cannot run game with visualization disabled")

        while self.pygame_manager.running:
            self.pygame_manager.event_handler()
            self.update()
            self.pygame_manager.draw(all_sprites_list=self.pygame_manager.get_all_sprites(), background_image_path=None)
            self.pygame_manager.tick(60)  # Limit to 60 FPS
        self.quit()

    def update(self):
        """Update game state"""
        self.pygame_manager.update(self.level, self.agent)

    def restart_level(self):
        """
        Restart the current level by resetting the agent's position and other necessary states.
        """
        print("Restarting level...")
        self.agent.rect.x = 50
        self.agent.rect.y = SCREEN_HEIGHT - 80
        self.agent.change_x = 0
        self.agent.change_y = 0

    def level_completed(self):
        """
        Handles the completion of a level in the game.
        """
        if self.training_mode:
            return

        print(f"Level {self.level_number} Completed!")

        if self.level_number < self.max_levels:
            self.level_number += 1
            print(f"Loading level {self.level_number}...")
            self.load_level()

            # Update PygameManager with new level
            self.pygame_manager.level = self.level
            self.pygame_manager.clear_sprites()
            self.pygame_manager.add_sprites(self.level.get_all_sprites())
            self.pygame_manager.add_sprites(self.agent)
            self.pygame_manager.set_blocks(self.level.get_blocks())

            # Update trainer with new level
            self.trainer.level = self.level
        else:
            print("Congratulations! You've completed all levels!")
            self.pygame_manager.running = False

    def quit(self):
        """
        Quits the game by shutting down the Pygame library and exiting the program.
        """
        pygame.quit()
        sys.exit()

    def load_level(self):
        """
        Load the current level based on the level number.
        """
        level_classes = {
            1: Level1,
            2: Level2,
            3: Level3,
            4: Level4,
            5: Level5,
            6: Level6,
            7: Level7,
            8: Level8
        }

        level_class = level_classes.get(self.level_number, Level1)
        self.level = level_class()

    def train_ai(self, num_episodes):
        """
        Start AI training with the specified number of episodes
        """
        self.trainer.train_ai(num_episodes)

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
