# world/blocks/interactive_block.py

from world.components.blocks.block import Block
from abc import abstractmethod

class InteractiveBlock(Block):
    def __init__(self, x, y):
        super().__init__(x, y)

    @abstractmethod
    def interact(self, agent):
        pass

    def update(self):
        # Interactive blocks may need to update
        pass
