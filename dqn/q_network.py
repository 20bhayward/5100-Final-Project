# q_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Dueling DQN architecture for more efficient learning.
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Increased network size
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Advantage stream
        self.adv_fc1 = nn.Linear(256, 128)
        self.adv_fc2 = nn.Linear(128, action_dim)
        
        # Value stream
        self.val_fc1 = nn.Linear(256, 128)
        self.val_fc2 = nn.Linear(128, 1)

        
    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Advantage stream
        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)
        
        # Value stream
        val = F.relu(self.val_fc1(x))
        val = self.val_fc2(val).expand(x.size(0), self.adv_fc2.out_features)
        
        # Combine streams
        q_vals = val + adv - adv.mean(1, keepdim=True).expand(x.size(0), self.adv_fc2.out_features)
        return q_vals
