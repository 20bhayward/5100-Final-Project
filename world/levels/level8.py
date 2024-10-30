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
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 220, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Add water below the starting platform
        for i in range(220, 2500, 40):
            water = WaterBlock(i, 560)  # Adjust y-coordinate for water below platform
            self.trap_list.add(water)
            self.all_sprites_list.add(water)

        # Vertical moving trap
        trap1 = MovingTrap(
            x=270, y=300, width=40, height=40, color=(255, 0, 0),
            speed=2, direction='vertical', start_pos=300, end_pos=560
        )
        self.trap_list.add(trap1)
        self.all_sprites_list.add(trap1)

        # Vertical moving trap
        trap2 = MovingTrap(
            x=330, y=360, width=40, height=40, color=(255, 0, 0),
            speed=1, direction='vertical', start_pos=360, end_pos=560
        )
        self.trap_list.add(trap2)
        self.all_sprites_list.add(trap2)

        # First staggered platform set
        platforms = [
            (400, 500),  # Step right
            (480, 430),  # Step left
            (560, 360),  # Step right
            (640, 290),  # Step left
        ]
        
        for x, y in platforms:
            block = SquareBlock(x, y)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        moving_platform1 = MovingBlock(
            x=720, y=250, width=60, height=20, color=(255, 255, 255),
            speed=1, direction='horizontal', start_pos=720, end_pos=850
        )
        self.block_list.add(moving_platform1)
        self.all_sprites_list.add(moving_platform1)

        # Second main platform
        for i in range(900, 1400, 40):
            block = SquareBlock(i, 200)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        trap4 = MovingTrap(
            x=1350, y=75, width=30, height=30, color=(255, 0, 0),
            speed=2, direction='horizontal', start_pos=1350, end_pos=1500  # Right-to-left movement
        )
        self.trap_list.add(trap4)
        self.all_sprites_list.add(trap4)

        # Wall
        x_position = 1400  # x-coordinate of the wall
        for y_position in range(200, 150, -40):  # Adjust range for desired wall height
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
