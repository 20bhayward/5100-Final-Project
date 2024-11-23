# world/levels/level4.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.trap_block import TrapBlock

class Level4(Level):
    def __init__(self):
        self.width = 1800  # Slightly reduced the level width
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 280, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # First platform with reduced traps
        for i in range(320, 520, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)
        # Removed the first trap to make it easier

        # Normal obstacle platform area
        for i in range(560, 760, 40):
            block = SquareBlock(i, 480)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second platform with staggered height
        platforms = [
            (840, 480),
            (920, 440),
            (1000, 400),
        ]

        for x, y in platforms:
            block = SquareBlock(x, y)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Removed one of the staggered platforms to simplify jumps

        # Moving platform with slower speed
        moving_platform = MovingBlock(
            x=1080,
            y=360,
            width=80,
            height=20,
            color=(255, 255, 255),
            speed=1,  # Reduced speed from 2 to 1
            direction='horizontal',
            start_pos=1080,
            end_pos=1280
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Elevated platforms with fewer traps
        for i in range(1300, 1500, 40):
            block = SquareBlock(i, 320)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Reduced the number of traps
        trap = TrapBlock(1360, 310)
        self.trap_list.add(trap)
        self.all_sprites_list.add(trap)

        # Final platform
        for i in range(1540, 1800, 40):
            block = SquareBlock(i, 440)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1760, 400)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
