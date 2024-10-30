# world/levels/level5.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level5(Level):
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

        # Corridor with spaced moving traps and platforms
        for i in range(250, 700, 200):  # Adding more spacing between moving traps
            trap = MovingBlock(
                x=i, y=500, width=40, height=40, color=(255, 0, 0),
                speed=2, direction='vertical', start_pos=480, end_pos=600
            )
            self.trap_list.add(trap)
            self.all_sprites_list.add(trap)

            # Platform above each moving trap
            platform = SquareBlock(i + 100, 550)  # Slightly above the trap's path
            self.block_list.add(platform)
            self.all_sprites_list.add(platform)

        # Second platform section for agent navigation
        for i in range(900, 900, 40):
            block = SquareBlock(i, 480)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        moving_platform = MovingBlock(
            x=850, y=500, width=80, height=20, color=(255, 255, 255),
            speed=2, direction='horizontal', start_pos=800, end_pos=1000
        )
        self.block_list.add(moving_platform)
        self.all_sprites_list.add(moving_platform)

        # Final platform with goal
        for i in range(1000, 1380, 40):
            block = SquareBlock(i, 440)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        goal_block = GoalBlock(1330, 400)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)