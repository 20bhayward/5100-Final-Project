# world/levels/level1.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.goal_block import GoalBlock


class Level1(Level):
    def __init__(self):
        self.width = 1200  # Set specific width for level 1
        self.height = 600
        super().__init__()

    def create_level(self):
        # Create ground using SquareBlocks
        for i in range(0, 800, 40):  # Assuming screen width is 800
            square_block = SquareBlock(i, 560)
            self.block_list.add(square_block)
            self.all_sprites_list.add(square_block)

        # Place the goal at the end of the level
        goal_block = GoalBlock(760, 520)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
