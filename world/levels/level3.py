# world/levels/level3.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.trap_block import TrapBlock

class Level3(Level):
    def __init__(self):
        self.width = 1400  # Shortened the level width
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 240, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # First jump platform
        for i in range(320, 440, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second platform with slight elevation
        for i in range(520, 640, 40):
            block = SquareBlock(i, 520)  # Slightly higher
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Third platform introducing a single trap
        for i in range(720, 900, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Single trap
        trap = TrapBlock(800, 500)
        self.trap_list.add(trap)
        self.all_sprites_list.add(trap)

        # Final stretch towards the goal
        for i in range(1000, 1200, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1160, 480)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
