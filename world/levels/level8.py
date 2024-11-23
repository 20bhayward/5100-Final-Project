# world/levels/level8.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.water_block import WaterBlock

class Level8(Level):
    def __init__(self):
        self.width = 2000  # Reduced level width
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 220, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Water hazard
        for i in range(240, 2000, 40):
            water = WaterBlock(i, 520)
            self.all_sprites_list.add(water)

        # First staggered platform set
        platforms = [
            (320, 570),
            (400, 520),
            (480, 470),
            (560, 420),
            (640, 370),
            (720, 320),
        ]
        for x, y in platforms:
            block = SquareBlock(x, y)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Moving platform
        moving_platform1 = MovingBlock(
            x=770, y=270, width=60, height=20, color=(255, 255, 255),
            speed=1, direction='horizontal', start_pos=770, end_pos=870
        )
        self.block_list.add(moving_platform1)
        self.all_sprites_list.add(moving_platform1)

        # Second main platform
        for i in range(900, 1400, 40):
            block = SquareBlock(i, 250)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Removed some traps and obstacles to shorten the level

        # Final platform towards the goal
        for i in range(1500, 1800, 40):
            block = SquareBlock(i, 300)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1760, 260)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
