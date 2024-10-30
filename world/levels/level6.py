# world/levels/level7.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.water_block import WaterBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.moving_trap import MovingTrap
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level6(Level):
    def __init__(self):
        self.width = 2000
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 200, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Add water
        for i in range(200, 450, 40):
            water = WaterBlock(i, 520)  # Adjust y-coordinate for water below platform
            self.trap_list.add(water)
            self.all_sprites_list.add(water)

        for i in range(250, 300, 120):
            block = SquareBlock(i, 480)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        for i in range(350, 400, 40):
            block = SquareBlock(i, 400)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Wall
        x_position = 500  # x-coordinate of the wall
        for y_position in range(560, 300, -40):  # Adjust range for desired wall height
            block = SquareBlock(x_position, y_position)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Add water
        for i in range(540, 2000, 40):
            water = WaterBlock(i, 350, height=400)  # Adjust y-coordinate for water below platform
            self.trap_list.add(water)
            self.all_sprites_list.add(water)

        # Second platform for movement
        for i in range(550, 900, 40):  # Shorter second platform
            block = SquareBlock(i, 250)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Moving platform
        moving_platform = MovingBlock(
            x=1000, y=380, width=80, height=20, color=(255, 255, 255),
            speed=2, direction='horizontal', start_pos=1000, end_pos=1400
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Traps
        trap1 = MovingTrap(
            x=900, y=330, width=40, height=40, color=(255, 0, 0),
            speed=1, direction='horizontal', start_pos=900, end_pos=1400
        )
        self.trap_list.add(trap1)
        self.all_sprites_list.add(trap1)

        # 3rd Platform
        for i in range(1500, 1800, 40):
            block = SquareBlock(i, 440)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1700, 400)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)