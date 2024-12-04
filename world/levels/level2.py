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

        # First gap with safe landing
        for i in range(300, 520, 40): 
            block = SquareBlock(i, 560)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Small elevation change
        for i in range(575, 800, 40):
            block = SquareBlock(i, 520) 
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Platform with first trap
        for i in range(875, 1100, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Final stretch - small gap then goal
        for i in range(1200, 1400, 40):
            block = SquareBlock(i, 520)
            self.block_list.add(block)
            self.all_sprites_list.add(block)

        # Goal at the end
        goal_block = GoalBlock(1340, 480)
        self.block_list.add(goal_block)
        self.goal_list.add(goal_block)
        self.all_sprites_list.add(goal_block)