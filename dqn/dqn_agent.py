# dqn_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .q_network import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        
        # Initialize networks with new architecture
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Improved optimizer configuration
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
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.epsilon_decay_steps = 0
        self.adaptive_epsilon = True
        
        # Training parameters
        self.gamma = 0.99
        self.tau = 0.001
        self.clip_grad_norm = 1.0
        
        # Performance tracking
        self.training_rewards = []
        self.training_losses = []
        self.mean_q_values = []
        
    def decay_epsilon(self):
        """Standard epsilon decay method for compatibility."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def choose_action(self, state, evaluation=False):
        """Choose action using epsilon-greedy policy with adaptive exploration."""
        if not evaluation and np.random.random() < self.epsilon:
            # During exploration, occasionally prefer actions that make physical sense
            if np.random.random() < 0.3:  # 30% of exploration actions
                return self._choose_heuristic_action(state)
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            self.mean_q_values.append(q_values.mean().item())
            return q_values.argmax().item()
    
    def _choose_heuristic_action(self, state):
        """Choose action based on simple heuristics for better exploration."""
        # Extract relevant features from state using correct indices
        on_platform = state[0]  # First element is on_platform
        distance_to_edge = state[5]  # Distance to platform edge
        nearest_gap_dist = state[6]  # Distance to nearest gap
        gap_jumpable = state[8]  # Whether gap is jumpable
        can_jump = state[19]  # Can jump status
        
        # Don't walk off the edge unless there's a jumpable gap
        if on_platform and distance_to_edge < 0.2:  # Getting close to edge
            if gap_jumpable > 0 and can_jump:
                return 1  # Jump
            return 2  # Stop
            
        if gap_jumpable > 0 and nearest_gap_dist < 0.3 and can_jump:
            return 1  # Jump action when approaching jumpable gap
        elif not on_platform:
            return 0  # Move right when in air
        else:
            return 0  # Otherwise, keep moving right
    
    def train(self, states, actions, rewards, next_states, dones):
        """Train the network using experience replay."""
        try:
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            # Get current Q values
            current_q_values = self.q_network(states)
            current_q_values = current_q_values.gather(1, actions)
            
            # Compute target Q values using Double DQN
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states)
                next_q_values = next_q_values.gather(1, next_actions)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute Huber loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Optimize the network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            
            # Store loss for monitoring
            self.training_losses.append(loss.item())

            
            return loss.item()
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return None
    
    # def _update_epsilon(self):
    #     """Update epsilon with adaptive decay based on performance."""
    #     if len(self.training_rewards) >= 10:
    #         recent_rewards = self.training_rewards[-10:]
    #         avg_reward = np.mean(recent_rewards)
            
    #         if avg_reward > 100:  # Good performance
    #             self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
    #         else:  # Poor performance
    #             self.epsilon = min(1.0, self.epsilon / 0.995)
    #     else:
    #         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
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
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses,
            'mean_q_values': self.mean_q_values
        }, filename)
    
    def load(self, filename):
        """Load the model and training state."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
        self.mean_q_values = checkpoint.get('mean_q_values', [])