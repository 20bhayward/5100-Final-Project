# world/levels/level3.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.trap_block import TrapBlock

class Level3(Level):
    def __init__(self):
        self.width = 1400
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 240, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # First jump platform
        for i in range(280, 440, 40):
            block = SquareBlock(i, 550)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second platform with slight elevation
        for i in range(480, 640, 40):
            block = SquareBlock(i, 530)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Third platform introducing a single trap
        for i in range(680, 900, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Final stretch towards the goal
        for i in range(960, 1200, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1160, 480)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
