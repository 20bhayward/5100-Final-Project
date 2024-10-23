# world/blocks/static_block.py

from world.components.blocks.block import Block

class StaticBlock(Block):
    def __init__(self, x, y):
        super().__init__(x, y)

    def update(self):
        # Static blocks do not need to update
        pass
