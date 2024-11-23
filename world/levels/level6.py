# world/levels/level6.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.water_block import WaterBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level6(Level):
    def __init__(self):
        self.width = 1800  # Adjusted level width
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 200, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Water hazard
        for i in range(200, 540, 40):
            water = WaterBlock(i, 520)
            self.all_sprites_list.add(water)

        # Series of platforms
        platform_positions = [
            (250, 520),
            (320, 470),
            (390, 420),
            (440, 370)
        ]

        for x, y in platform_positions:
            block = SquareBlock(x, y)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Wall to challenge the player
        x_position = 500
        for y_position in range(560, 300, -40):
            block = SquareBlock(x_position, y_position)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Water after the wall
        for i in range(540, 1800, 40):
            water = WaterBlock(i, 350, height=400)
            self.trap_list.add(water)
            self.all_sprites_list.add(water)

        # Second platform
        for i in range(570, 900, 40):
            block = SquareBlock(i, 280)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Moving platform with reduced speed
        moving_platform = MovingBlock(
            x=1000, y=380, width=120, height=20, color=(255, 255, 255),
            speed=1.5,  # Reduced speed from 2 to 1
            direction='horizontal',
            start_pos=1000,
            end_pos=1400
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Final platform
        for i in range(1500, 1800, 40):
            block = SquareBlock(i, 440)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1700, 400)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
