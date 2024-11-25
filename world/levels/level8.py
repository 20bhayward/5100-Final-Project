# world/levels/level8.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.trap_block import TrapBlock
from world.components.blocks.interactive.goal_block import GoalBlock

class Level8(Level):
    def __init__(self):
        self.width = 2000
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform - wide for preparation
        for i in range(0, 320, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # First challenge: Wide gap with trap
        for i in range(440, 680, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Trap in the gap
        for i in range(320, 440, 40):
            trap = TrapBlock(i, 580, 40, 40)
            self.trap_list.add(trap)
            self.all_sprites_list.add(trap)

        # Second challenge: Moving platform sequence
        # First moving platform
        moving_platform1 = MovingBlock(
            x=720,
            y=560,
            width=120,  # Wide enough for stable landing
            height=40,
            color=(255, 255, 255),
            speed=1.5,
            direction='horizontal',
            start_pos=720,
            end_pos=920
        )
        self.block_list.add(moving_platform1)
        self.all_sprites_list.add(moving_platform1)

        # Third challenge: Elevated section with precise jumps
        for i in range(960, 1160, 40):
            block = SquareBlock(i, 520)  # Slight elevation
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Single block jumps with traps (testing precise movement)
        platform_positions = [
            (1240, 520, 120),  # (x, y, width) - Wide landing platform
            (1440, 520, 120),  # Wide landing platform
            (1640, 520, 120),  # Wide landing platform
        ]

        for x, y, width in platform_positions:
            for w in range(0, width, 40):
                block = SquareBlock(x + w, y)
                self.block_list.add(block)
                self.all_sprites_list.add(block)

        # Traps between platforms
        trap_sections = [
            (1160, 1240),
            (1360, 1440),
            (1560, 1640)
        ]

        for start, end in trap_sections:
            for i in range(start, end, 40):
                trap = TrapBlock(i, 580, 40, 40)
                self.trap_list.add(trap)
                self.all_sprites_list.add(trap)

        # Final moving platform to goal
        moving_platform2 = MovingBlock(
            x=1760,
            y=520,
            width=120,
            height=40,
            color=(255, 255, 255),
            speed=1,
            direction='horizontal',
            start_pos=1760,
            end_pos=1900
        )
        self.block_list.add(moving_platform2)
        self.all_sprites_list.add(moving_platform2)

        # Final platform and goal
        for i in range(1940, 2000, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal block
        goal_block = GoalBlock(1960, 480)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)