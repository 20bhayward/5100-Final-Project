# dqn_agent.py

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from q_network import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()

        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration probability
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.device = device

        # Update target network weights to match the main Q network
        self.update_target_network()

    def choose_action(self, state, epsilon=0.999):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.q_network.fc3.out_features)
        else:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self, replay_buffer, batch_size):
        if replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute the Q-values for the current states
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the Q-values for the next states
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]

        # Compute the target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Update the Q-network using the loss
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        # Save the Q network's state dictionary
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        # Load the Q network's state dictionary
        self.q_network.load_state_dict(torch.load(filename))
        self.q_network.eval()  # Set the model to evaluation mode

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
