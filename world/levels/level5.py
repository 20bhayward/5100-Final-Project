# world/levels/level5.py

import pygame
from world.components.blocks.interactive.trap_block import TrapBlock
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.water_block import WaterBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level5(Level):
    def __init__(self):
        self.width = 1600
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 200, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Water hazard
        for i in range(200, 1600, 40):
            water = WaterBlock(i, 530)
            self.all_sprites_list.add(water)

        # Platforms without moving traps
        for i in range(220, 700, 150):
            # Platform above the water
            platform = SquareBlock(i, 540)
            self.block_list.add(platform)
            self.all_sprites_list.add(platform)
        
        # Platforms without moving traps
        for i in range(220, 700, 150):
            # Platform above the water
            platform = SquareBlock(i + 40, 540)
            self.block_list.add(platform)
            self.all_sprites_list.add(platform)

        # Moving platform
        moving_platform = MovingBlock(
            x=750, y=500, width=80, height=20, color=(255, 255, 255),
            speed=1, direction='horizontal', start_pos=700, end_pos=1000
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Final platform with goal
        for i in range(1000, 1400, 40):
            block = SquareBlock(i, 500)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        goal_block = GoalBlock(1360, 460)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
