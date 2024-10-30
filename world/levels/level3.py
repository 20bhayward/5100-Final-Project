# world/levels/level3.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.static.rectangle_block import RectangleBlock
from world.components.blocks.interactive.water_block import WaterBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.trap_block import TrapBlock

class Level3(Level):
    def __init__(self):
        self.width = 1800
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

        # First basic jump
        for i in range(320, 440, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second platform with slight elevation
        for i in range(520, 640, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Third platform introducing single trap
        for i in range(720, 880, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)
        
        # Single trap to introduce hazards
        trap = TrapBlock(800, 480)
        self.trap_list.add(trap)
        self.all_sprites_list.add(trap)

        # Vertical section with staggered platforms
        platforms = [
            (960, 520),   # Start of vertical section
            (1040, 460),  # First step up
            (960, 400),   # Second step up
            (1040, 340),  # Third step up
            (960, 280),   # Final platform
        ]
        
        for x, y in platforms:
            for i in range(x, x + 40, 40):  # Each platform is 3 blocks wide
                block = SquareBlock(i, y)
                self.block_list.add(block)
                self.all_sprites_list.add(block)

        # Upper section with traps
        for i in range(1120, 1440, 40):
            block = SquareBlock(i, 280)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Traps on upper section with safe spots
        for i in range(1160, 1400, 120):
            trap = TrapBlock(i, 240)
            self.trap_list.add(trap)
            self.all_sprites_list.add(trap)

        # Moving platform
        moving_platform = MovingBlock(
            x=1480,
            y=280,
            width=80,
            height=20,
            color=(255, 255, 255),
            speed=2,
            direction='vertical',
            start_pos=280,
            end_pos=440
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Final platform
        for i in range(1600, 1800, 40):
            block = SquareBlock(i, 440)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Final traps
        for i in range(1640, 1760, 120):
            trap = TrapBlock(i, 400)
            self.trap_list.add(trap)
            self.all_sprites_list.add(trap)

        # Goal
        goal_block = GoalBlock(1720, 400)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)