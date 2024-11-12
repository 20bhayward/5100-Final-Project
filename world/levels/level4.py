# world/levels/level4.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.static.rectangle_block import RectangleBlock
from world.components.blocks.interactive.water_block import WaterBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.trap_block import TrapBlock

class Level4(Level):
    def __init__(self):
        self.width = 2000
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 240, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Add water
        for i in range(240, 2000, 40):
            water = WaterBlock(i, 520)
            self.trap_list.add(water)
            self.all_sprites_list.add(water)

        # First challenging section with traps
        for i in range(320, 500, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)
        trap = TrapBlock(400, 490)
        self.trap_list.add(trap)
        self.all_sprites_list.add(trap)

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
            (1080, 360)
        ]
        
        for x, y in platforms:
            block = SquareBlock(x, y)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Moving platform to reach the next section
        moving_platform = MovingBlock(
            x=1160,
            y=360,
            width=80,
            height=20,
            color=(255, 255, 255),
            speed=2,
            direction='horizontal',
            start_pos=1160,
            end_pos=1370
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Elevated platforms with traps
        for i in range(1380, 1600, 40):
            block = SquareBlock(i, 320)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        for i in range(1460, 1560, 100):
            trap = TrapBlock(i, 310)
            self.trap_list.add(trap)
            self.all_sprites_list.add(trap)

        # Final platform
        for i in range(1680, 2000, 40):
            block = SquareBlock(i, 440)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Final goal section
        goal_block = GoalBlock(1920, 400)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)