# world/levels/level7.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level7(Level):
    def __init__(self):
        self.width = 1600
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform (wide for good start)
        for i in range(0, 280, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # First section with small gap and trap
        for i in range(360, 600, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # First trap covering the gap
        for i in range(280, 360, 40):
            trap = TrapBlock(i, 580, 40, 40)
            self.trap_list.add(trap)
            self.all_sprites_list.add(trap)

        # Second section with medium gap
        for i in range(680, 920, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second trap covering the wider gap
        for i in range(600, 680, 40):
            trap = TrapBlock(i, 580, 40, 40)
            self.trap_list.add(trap)
            self.all_sprites_list.add(trap)

        # Moving platform section
        moving_platform = MovingBlock(
            x=1000,
            y=560,
            width=160,
            height=40,
            color=(255, 255, 255),
            speed=1,
            direction='horizontal',
            start_pos=950,
            end_pos=1250
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Final platform sequence
        for i in range(1280, 1600, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1540, 520)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)