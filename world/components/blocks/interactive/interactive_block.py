# world/blocks/interactive_block.py

from world.components.blocks.block import Block
from abc import abstractmethod

class InteractiveBlock(Block):
    """
    InteractiveBlock is a subclass of Block that represents a block which can interact with an agent.

    Methods
    -------
    __init__(x, y)
        Initializes the InteractiveBlock with the given x and y coordinates.

    interact(agent)
        Abstract method that defines how the block interacts with an agent. Must be implemented by subclasses.

    update()
        Updates the state of the InteractiveBlock. This method can be overridden by subclasses if needed.
    """
    def __init__(self, x, y):
        """
        Initializes an InteractiveBlock instance with the given x and y coordinates.

        Args:
            x (int): The x-coordinate of the block.
            y (int): The y-coordinate of the block.
        """
        super().__init__(x, y)

    @abstractmethod
    def interact(self, agent):
        """
        Defines the interaction behavior for the block when an agent interacts with it.

        Parameters:
        agent (object): The agent that is interacting with the block.

        Returns:
        None
        """
        pass

    def update(self):
        """
        Update the state of the interactive block.

        This method is intended to be overridden by subclasses to provide
        specific update functionality for different types of interactive blocks.
        """
        # Interactive blocks may need to update
        pass
