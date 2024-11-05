import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the DQN model
class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.

    This model consists of three convolutional layers followed by batch normalization and a fully connected layer.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        bn3 (nn.BatchNorm2d): Batch normalization for the third convolutional layer.
        head (nn.Linear): Fully connected layer.

    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the network.
    """
    def __init__(self, h, w, outputs):


        """
        Initializes the DQN model.

        Args:
            h (int): The height of the input image.
            w (int): The width of the input image.
            outputs (int): The number of output actions.

        Attributes:
            conv1 (nn.Conv2d): First convolutional layer.
            bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
            conv2 (nn.Conv2d): Second convolutional layer.
            bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
            conv3 (nn.Conv2d): Third convolutional layer.
            bn3 (nn.BatchNorm2d): Batch normalization for the third convolutional layer.
            head (nn.Linear): Fully connected layer that outputs the final Q-values.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            """
            Calculate the output size of a convolutional layer.

            Args:
                size (int): The size of the input (height or width).
                kernel_size (int, optional): The size of the kernel. Default is 5.
                stride (int, optional): The stride of the convolution. Default is 2.

            Returns:
                int: The size of the output after applying the convolution.
            """
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
