# world/levels/level6.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level6(Level):
    def __init__(self):
        self.width = 2000  # Extended level width for dual paths
        self.height = 800
        super().__init__()

    def create_level(self):
        # === STARTING PLATFORM ===
        for i in range(0, 400, 40):  # Wide starting platform
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # First straight section with a small gap
        for i in range(480, 1000, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second section after gap
        for i in range(1080, 1700, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        for i in range(480, 680, 40):
            block = SquareBlock(i, 520) 
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second elevated platform (similar to Level 4's first elevation)
        for i in range(760, 960, 40):
            block = SquareBlock(i, 480)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Third elevated platform
        for i in range(1040, 1240, 40):
            block = SquareBlock(i, 440) 
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Moving platform sequence (upper path)
        moving_platform2 = MovingBlock(
            x=1320,
            y=440,
            width=160,
            height=40,
            color=(255, 255, 255),
            speed=1.5,
            direction='horizontal',
            start_pos=1320,
            end_pos=1520
        )
        self.block_list.add(moving_platform2)
        self.all_sprites_list.add(moving_platform2)

        # Final upper platform (gentle slope down to goal)
        for i in range(1600, 1800, 40):
            block = SquareBlock(i, 480)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Final platform
        for i in range(1800, 2000, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1950, 520)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)