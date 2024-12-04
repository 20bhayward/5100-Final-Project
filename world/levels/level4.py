# world/levels/level4.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.trap_block import TrapBlock

class Level4(Level):
    def __init__(self):
        self.width = 1800
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 280, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # First platform
        for i in range(320, 520, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second platform
        for i in range(580, 780, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Third platform
        for i in range(840, 1040, 40):
            block = SquareBlock(i, 480)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Fourth platform
        for i in range(1100, 1300, 40):
            block = SquareBlock(i, 440)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Moving platform
        moving_platform = MovingBlock(
            x=1360,
            y=440,
            width=160,
            height=30,
            color=(255, 255, 255),
            speed=1,
            direction='horizontal',
            start_pos=1360,
            end_pos=1560
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Fifth platform
        for i in range(1620, 1820, 40):
            block = SquareBlock(i, 400)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1780, 360)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)