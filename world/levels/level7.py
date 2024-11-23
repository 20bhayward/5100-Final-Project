# world/levels/level7.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.water_block import WaterBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level7(Level):
    def __init__(self):
        self.width = 1600  # Reduced level width
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 180, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Water hazard
        for i in range(200, 740, 40):
            water = WaterBlock(i, 520)
            self.all_sprites_list.add(water)

        # Staggered platforms
        platforms = [
            (230, 520),
            (310, 470),
            (390, 420),
            (470, 370),
            (550, 320),
            (630, 270),
            (710, 220),
        ]
        for x, y in platforms:
            block = SquareBlock(x, y)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Wall
        x_position = 790
        for y_position in range(560, 180, -40):
            block = SquareBlock(x_position, y_position)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Second moving platform without traps
        moving_platform = MovingBlock(
            x=1030, y=360, width=80, height=20, color=(255, 255, 255),
            speed=2, direction='horizontal', start_pos=980, end_pos=1400
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Third platform
        for i in range(1500, 1600, 40):
            block = SquareBlock(i, 480)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1560, 440)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
