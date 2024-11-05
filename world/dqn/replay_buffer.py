import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    ReplayBuffer is a class that implements a replay buffer for storing and sampling experience tuples in reinforcement learning.

    Attributes:
        buffer (deque): A deque to store experience tuples with a maximum length of buffer_size.
        batch_size (int): The number of experience tuples to sample from the buffer.

    Methods:
        __init__(buffer_size, batch_size):
            Initializes the ReplayBuffer with a specified buffer size and batch size.
        
        store(state, action, reward, next_state, done):
            Stores an experience tuple in the buffer.
        
        sample():
            Samples a batch of experience tuples from the buffer.
        
        size():
            Returns the current size of the buffer.
    """
    def __init__(self, buffer_size, batch_size):
        """
        Initializes the ReplayBuffer with a specified buffer size and batch size.

        Args:
            buffer_size (int): The maximum size of the buffer.
            batch_size (int): The size of the batches to sample from the buffer.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_state: The state of the environment after taking the action.
            done: A boolean indicating whether the episode has ended.

        Returns:
            None
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly samples a batch of experiences from the replay buffer.

        Returns:
            tuple: A tuple containing the following elements:
                - states (np.ndarray): Array of states.
                - actions (np.ndarray): Array of actions taken.
                - rewards (np.ndarray): Array of rewards received.
                - next_states (np.ndarray): Array of next states.
                - dones (np.ndarray): Array of done flags indicating if the episode has ended.
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of elements currently stored in the buffer.
        """
        return len(self.buffer)
