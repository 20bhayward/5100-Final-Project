# world/levels/level7.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.water_block import WaterBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.moving_trap import MovingTrap
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level7(Level):
    def __init__(self):
        self.width = 2000
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 180, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Add water below the starting platform
        for i in range(200, 740, 40):
            water = WaterBlock(i, 520)  # Adjust y-coordinate for water below platform
            self.trap_list.add(water)
            self.all_sprites_list.add(water)

        # Vertical section with staggered platforms
        platforms = [
            (230, 520),   # First step up, right
            (310, 470),   # Step up, right
            (390, 420),   # Step up, right
            (470, 370),   # Step up, right
            (550, 320),   # Step up, right
            (630, 270),   # Step up, right
            (710, 220),   # Final step up, right
        ] 
        for x, y in platforms:
            block = SquareBlock(x, y)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Wall
        x_position = 790  # x-coordinate of the wall
        while x_position < 1000:
            for y_position in range(560, 180, -40):  # Adjust range for desired wall height
                block = SquareBlock(x_position, y_position)
                self.block_list.add(block)
                self.all_sprites_list.add(block)
            x_position += 40

        # Add water below the second platform
        for i in range(1030, 2000, 40):
            water = WaterBlock(i, 520)  # Adjust y-coordinate for water below platform
            self.trap_list.add(water)
            self.all_sprites_list.add(water)

        # First moving trap between the second platform and the moving platform
        trap1 = MovingTrap(
            x=1030, y=300, width=40, height=40, color=(255, 0, 0),
            speed=1, direction='horizontal', start_pos=1030, end_pos=1500
        )
        self.trap_list.add(trap1)
        self.all_sprites_list.add(trap1)

        moving_platform = MovingBlock(
            x=1030, y=360, width=80, height=20, color=(255, 255, 255),
            speed=2, direction='horizontal', start_pos=1030, end_pos=1500
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        trap2 = MovingTrap(
            x=1030, y=380, width=40, height=40, color=(255, 0, 0),
            speed=1, direction='horizontal', start_pos=1030, end_pos=1500
        )
        self.trap_list.add(trap2)
        self.all_sprites_list.add(trap2)

        # Third platform section for agent navigation
        for i in range(1500, 1700, 40):
            block = SquareBlock(i, 480)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        trap = TrapBlock(1550, 440)
        self.trap_list.add(trap)
        self.all_sprites_list.add(trap)

        goal_block = GoalBlock(1650, 440)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)