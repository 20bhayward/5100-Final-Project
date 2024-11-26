import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .q_network import QNetwork
from core.config import MOVEMENT_ACTIONS

class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim

        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer configuration
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=0.0003,
            weight_decay=0.01,
            amsgrad=True
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Simple exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Training parameters
        self.gamma = 0.99
        self.tau = 0.001
        self.clip_grad_norm = 1.0

        # Performance tracking
        self.training_losses = []
        self.mean_q_values = []

    def decay_epsilon(self):
        """Simple epsilon decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self, state, evaluation=False):
        """Choose action using epsilon-greedy policy."""
        if not evaluation and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train(self, states, actions, rewards, next_states, dones):
        """Train the network using experience replay."""
        try:
            # 1. Convert inputs to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # 2. Calculate current Q-values
            current_q_values = self.q_network(states)
            current_q_values = current_q_values.gather(1, actions)

            # 3. Calculate target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # 4. Compute loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # 5. Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            # 6. Store metrics
            self.training_losses.append(loss.item())
            return loss.item()

        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None

    def update_target_network(self):
        """Soft update of target network."""
        for target_param, local_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data +
                (1.0 - self.tau) * target_param.data
            )

    def save(self, filename):
        """Save the model and training state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_losses': self.training_losses,
            'mean_q_values': self.mean_q_values
        }, filename)

    def load(self, filename):
        """Load the model and training state."""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_losses = checkpoint['training_losses']
            self.mean_q_values = checkpoint['mean_q_values']
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
