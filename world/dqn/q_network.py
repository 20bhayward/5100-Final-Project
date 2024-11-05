import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    QNetwork is a neural network model for approximating the Q-value function in reinforcement learning.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action space.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.

    Methods:
        forward(state):
            Performs a forward pass through the network.
            Args:
                state (torch.Tensor): Input state tensor.
            Returns:
                torch.Tensor: Output action-value tensor.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the QNetwork.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output actions.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        """
        Perform a forward pass through the neural network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)