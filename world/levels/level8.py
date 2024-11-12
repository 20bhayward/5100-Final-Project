# world/levels/level8.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.moving_trap import MovingTrap
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.water_block import WaterBlock

class Level8(Level):
    def __init__(self):
        self.width = 2500
        self.height = 800
        self.water_blocks = []
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 220, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Add water below the starting platform
        for i in range(240, 2500, 40):
            water = WaterBlock(i, 520)  # Adjust y-coordinate for water below platform
            self.trap_list.add(water)
            self.all_sprites_list.add(water)
            self.water_blocks.append(water)

        # Vertical moving trap
        trap1 = MovingTrap(
            x=270, y=300, width=40, height=40, color=(255, 0, 0),
            speed=2, direction='vertical', start_pos=300, end_pos=560
        )
        self.trap_list.add(trap1)
        self.all_sprites_list.add(trap1)

        # Vertical moving trap
        trap1 = MovingTrap(
            x=450, y=400, width=20, height=20, color=(255, 0, 0),
            speed=2, direction='vertical', start_pos=400, end_pos=510
        )
        self.trap_list.add(trap1)
        self.all_sprites_list.add(trap1)

        # Vertical moving trap
        trap1 = MovingTrap(
            x=610, y=300, width=20, height=20, color=(255, 0, 0),
            speed=2, direction='vertical', start_pos=300, end_pos=410
        )
        self.trap_list.add(trap1)
        self.all_sprites_list.add(trap1)

        # First staggered platform set
        platforms = [
            (320, 570),  # Step right
            (400, 520),  # Step right
            (480, 470),  # Step right
            (560, 420),  # Step right
            (640, 370),  # Step right
            (720, 320),  # Step right
        ]
        
        for x, y in platforms:
            block = SquareBlock(x, y)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

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

        trap4 = MovingTrap(
            x=1350, y=110, width=30, height=30, color=(255, 0, 0),
            speed=1, direction='horizontal', start_pos=1350, end_pos=1500  # Right-to-left movement
        )
        self.trap_list.add(trap4)
        self.all_sprites_list.add(trap4)

        # Wall
        x_position = 1400  # x-coordinate of the wall
        for y_position in range(220, 180, -40):  # Adjust range for desired wall height
            block = SquareBlock(x_position, y_position)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Additional trap before third platform
        trap5 = MovingTrap(
            x=1440, y=250, width=40, height=40, color=(255, 0, 0),
            speed=2, direction='horizontal', start_pos=1440, end_pos=1600  # Right-to-left movement
        )
        self.trap_list.add(trap5)
        self.all_sprites_list.add(trap5)

        # Third platform with slightly elevated sections
        for i in range(1600, 1650, 40):
            block = SquareBlock(i, 300)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Final challenge section
        for i in range(1750, 2300, 40):
            block = SquareBlock(i, 300)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block at the end of the level
        goal_block = GoalBlock(1960, 260)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)
