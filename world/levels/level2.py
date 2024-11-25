# world/levels/level2.py

import pygame
from world.levels.level import Level
from world.components.blocks.static.square_block import SquareBlock
from world.components.blocks.static.rectangle_block import RectangleBlock
from world.components.blocks.interactive.water_block import WaterBlock
from world.components.blocks.interactive.moving_block import MovingBlock
from world.components.blocks.interactive.goal_block import GoalBlock
from world.components.blocks.interactive.trap_block import TrapBlock

class Level2(Level):
    def __init__(self):
        self.width = 1400
        self.height = 800
        super().__init__()

    def create_level(self):
        # Starting platform
        for i in range(0, 240, 40):
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # # Water
        # for i in range(240, 1800, 40):
        #     water = WaterBlock(i, 620)
        #     self.trap_list.add(water)
        #     self.all_sprites_list.add(water)

        # First gap with safe landing
        for i in range(300, 520, 40):  # Wider platform for safe landing
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Small elevation change
        for i in range(575, 800, 40):
            block = SquareBlock(i, 520)  # Slightly higher
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Platform with first trap
        for i in range(875, 1100, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)
        
        # # Single trap in a predictable spot
        # trap = TrapBlock(950, 500, 25, 25)
        # self.trap_list.add(trap)
        # self.all_sprites_list.add(trap)

        # Final stretch - small gap then goal
        for i in range(1200, 1400, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # trap1 = TrapBlock(1240, 500, 25, 25)
        # self.trap_list.add(trap1)
        # self.all_sprites_list.add(trap1)

        # Goal at the end
        goal_block = GoalBlock(1340, 480)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)