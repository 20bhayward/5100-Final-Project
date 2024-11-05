# AI Runner Game Documentation
## Version 1.0.0

# Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Game Architecture](#game-architecture)
4. [AI System](#ai-system)
5. [Development Guide](#development-guide)
6. [API Reference](#api-reference)
7. [Training Guide](#training-guide)
8. [Troubleshooting](#troubleshooting)

# Introduction

## Project Overview
AI Runner Game is a 2D platformer that implements evolutionary AI and deep reinforcement learning to create adaptive agents that learn to navigate through procedurally generated levels. The project combines traditional game development with modern AI techniques to create an engaging learning environment.

## Key Features
- Physics-based 2D platformer mechanics
- Deep Q-Network (DQN) implementation
- Procedurally generated levels
- Progressive difficulty scaling
- Real-time training visualization

## Technical Stack
- Python 3.x
- PyGame for game engine
- PyTorch for neural networks
- NumPy for numerical computations

# Installation

## System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- NVIDIA GPU recommended for training
- Operating System: Windows 10/11, macOS 10.15+, or Linux

## Dependencies Installation
```bash
# Clone the repository
git clone https://github.com/20bhayward/5100-Final-Project.git
cd 5100-Final-Project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

## Verification
```bash
# Run tests
python -m pytest tests/

# Start the game
python game.py
```

# Game Architecture

## Core Components

### Game Engine (game.py)
The main game engine handles:
- Game loop management
- State updates
- Rendering
- Input processing
- Physics calculations

```python
class Game:
    def __init__(self, manual_control=False, level_number=1, training_mode=False):
        # Initialize game components
        self.manual_control = manual_control
        self.level_number = level_number
        self.training_mode = training_mode
        # ... additional initialization

    def run(self):
        # Main game loop
        while self.running:
            self.events()
            self.update()
            self.draw()
```

### Agent System (agent.py)
Handles both player-controlled and AI agents:
- Movement mechanics
- Collision detection
- State management
- Input processing

```python
class Agent:
    def __init__(self, x, y, screen_height=600):
        self.width = 20
        self.height = 20
        # ... physics properties
        self.on_ground = False

    def update(self, blocks):
        self.apply_physics()
        self.handle_collisions(blocks)
```

### Level System (level.py)
Manages level creation and progression:
- Platform generation
- Obstacle placement
- Difficulty scaling
- Goal management

# AI System

## Deep Q-Network Implementation

### Network Architecture
```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
```

### Training Pipeline
```python
class DQNAgent:
    def train(self, replay_buffer, batch_size):
        # Sample transitions
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        
        # Compute Q values
        current_q = self.q_network(states).gather(1, actions)
        next_q = self.target_network(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Update network
        loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## State and Action Spaces

### State Space
- Agent position (x, y)
- Agent velocity (vx, vy)
- Distance to nearest obstacle
- Distance to goal
- Ground contact status
- Environmental features

### Action Space
- LEFT: Move left
- RIGHT: Move right
- JUMP: Perform jump
- NONE: No action

# Development Guide

## Code Organization
```
ai-runner-game/
├── agent/
│   ├── agent.py
│   └── dqn_agent.py
├── world/
│   ├── levels/
│   ├── components/
│   └── physics/
├── training/
│   ├── train_dqn.py
│   └── replay_buffer.py
├── utils/
└── game.py
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Include docstrings for classes and methods
- Maintain test coverage

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit pull request with detailed description

# API Reference

## Game Engine API
```python
class Game:
    """Main game engine class."""
    
    def __init__(self, manual_control=False, level_number=1, training_mode=False):
        """Initialize game instance."""
        pass

    def run(self):
        """Start the game loop."""
        pass

    def update(self):
        """Update game state."""
        pass
```

## Agent API
```python
class Agent:
    """Base agent class."""

    def update(self, blocks):
        """Update agent state."""
        pass

    def move(self, action):
        """Execute movement action."""
        pass
```

# Training Guide

## Basic Training
```python
# Start training
python train_dqn.py --episodes 1000 --batch-size 64

# Resume training
python train_dqn.py --load checkpoint.pth --episodes 500
```

## Hyperparameters
- Learning rate: 0.001
- Discount factor: 0.99
- Epsilon start: 1.0
- Epsilon end: 0.01
- Epsilon decay: 0.995
- Replay buffer size: 10000
- Batch size: 64
- Target update frequency: 10 episodes

## Performance Monitoring
```python
# Training metrics
episode_rewards = []
completion_times = []
average_q_values = []

# Visualization
plt.plot(episode_rewards)
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
```

# Troubleshooting

## Common Issues

### Installation Problems
```bash
# CUDA not found
pip install torch==2.4.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Pygame installation fails
pip install --upgrade pip setuptools wheel
pip install pygame
```

### Training Issues
- High memory usage: Reduce batch size
- Slow training: Enable GPU acceleration
- Poor performance: Adjust hyperparameters

### Runtime Issues
- Frame rate drops: Optimize rendering
- Physics glitches: Check collision detection
- Memory leaks: Monitor sprite cleanup

## Performance Optimization
- Use sprite groups for efficient rendering
- Implement object pooling for projectiles
- Optimize collision detection algorithms
- Enable GPU acceleration for training

# Future Development

## Planned Features
- Enhanced procedural generation
- Multiple training algorithms
- Advanced visualization tools
- Network architecture improvements

## Research Directions
- Multi-agent training
- Curriculum learning
- Meta-learning approaches
- Dynamic difficulty adjustment

# Version History

## v1.0.0
- Initial release
- Basic game mechanics
- DQN implementation
- Level generation system

This documentation is maintained by the project team and is updated regularly with new features and improvements.