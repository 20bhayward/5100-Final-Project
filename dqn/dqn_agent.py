import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .q_network import QNetwork

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent

    Attributes:
        device (torch.device): The device to run the model on (CPU or GPU).
        q_network (QNetwork): The Q-network model.
        target_network (QNetwork): The target Q-network model.
        optimizer (torch.optim.Optimizer): The optimizer for training the Q-network.
        action_dim (int): The dimension of the action space.
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        epsilon_min (float): The minimum exploration rate.
        epsilon_decay (float): The decay rate for the exploration rate.
        gamma (float): The discount factor for future rewards.

    Methods:
        choose_action(state):
            Selects an action based on the current state using an epsilon-greedy policy.

        train(states, actions, rewards, next_states, dones):
            Trains the Q-network using a batch of experience tuples.

        update_target_network():
            Updates the target network with the weights of the Q-network.

        decay_epsilon():
            Decays the exploration rate (epsilon) after each episode.

        save(filename):
            Saves the model and optimizer state to a file.

        load(filename):
            Loads the model and optimizer state from a file.
    """
    def __init__(self, state_dim, action_dim, device):
        """
        Initialize the DQNAgent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            device (torch.device): The device (CPU or GPU) to run the model on.

        Attributes:
            device (torch.device): The device (CPU or GPU) to run the model on.
            q_network (QNetwork): The Q-network used for action-value estimation.
            target_network (QNetwork): The target Q-network used for stable training.
            optimizer (torch.optim.Adam): The optimizer for training the Q-network.
            action_dim (int): Dimension of the action space.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate of exploration.
            gamma (float): Discount factor for future rewards.
        """
        self.device = device
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters())
        self.action_dim = action_dim

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor

    def choose_action(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy policy.

        Parameters:
        state (numpy.ndarray): The current state of the environment.

        Returns:
        int: The action chosen by the agent.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def train(self, states, actions, rewards, next_states, dones):
        """
        Train the DQN agent by performing a single step of gradient descent on the loss.

        Args:
            states (array-like): Batch of current states.
            actions (array-like): Batch of actions taken.
            rewards (array-like): Batch of rewards received.
            next_states (array-like): Batch of next states.
            dones (array-like): Batch of done flags indicating episode termination.

        Returns:
            float: The computed loss value.
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Get next Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Updates the target network by copying the weights from the Q-network.
        
        This method is typically used in deep Q-learning to periodically update the 
        target network with the weights of the Q-network to stabilize training.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """
        Decays the epsilon value by multiplying it with the epsilon decay rate.
        Ensures that epsilon does not fall below the minimum epsilon value.

        Attributes:
            epsilon (float): The current exploration rate.
            epsilon_min (float): The minimum exploration rate.
            epsilon_decay (float): The factor by which the exploration rate is decayed.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        """
        Save the current state of the DQN agent to a file.

        Parameters:
        filename (str): The path to the file where the model and optimizer states will be saved.

        The saved file will contain:
        - 'model_state_dict': The state dictionary of the Q-network.
        - 'optimizer_state_dict': The state dictionary of the optimizer.
        - 'epsilon': The current value of epsilon.

        Prints a message indicating the file to which the model was saved.
        """
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """
        Load the model parameters from a checkpoint file.

        Args:
            filename (str): The path to the checkpoint file.

        Loads the state dictionaries for the Q-network, target network, and optimizer
        from the checkpoint file. Also restores the epsilon value used for the epsilon-greedy policy.

        Prints a message indicating that the model has been loaded successfully.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filename}")
