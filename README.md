# AI Platform Game
## Project Overview

In this project, we've created a platform game where an AI agent learns to navigate through increasingly challenging levels. What makes this project unique is its focus on combining traditional gaming elements with reinforcement learning techniques. The agent must complete levels and perform actions that humans are capable of doing: timing jumps perfectly, avoiding hazards, and finding optimal paths to goals.

### Reinforcement Learning Implementation

Our implementation uses Deep Q-Learning (DQN)

#### Double DQN Architecture

We use two neural networks - a primary network for action selection and a target network for value estimation. This separation helps prevent the overestimation of Q-values, a common problem in Q-learning. The target network parameters are updated slowly through soft updates:

```python
def update_target_network(self):
    """Soft update of target network parameters."""
    for target_param, local_param in zip(
        self.target_network.parameters(),
        self.q_network.parameters()
    ):
        target_param.data.copy_(
            self.tau * local_param.data + 
            (1.0 - self.tau) * target_param.data
        )
```

#### Experience Replay System

Our experience replay buffer stores transitions (state, action, reward, next_state) and enables the agent to learn from past experiences:

```python
class ReplayBuffer:
    def store(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self):
        """Sample a batch of experiences for training."""
        batch = random.sample(self.buffer, self.batch_size)
        return map(np.array, zip(*batch))
```

### Perception System

The agent's perception system is designed to mimic how humans process visual information when playing platform games. It includes:

#### Camera-Based State Representation

1. Platform Layout
   - Current platform properties (width, stability)
   - Distance to platform edges
   - Gap locations and sizes
   - Height differences between platforms

2. Hazard Detection
   - Proximity to dangerous elements
   - Safe zones for landing
   - Trajectory analysis for jump safety

3. Goal Information
   - Direction to goal
   - Distance estimation
   - Potential path obstacles

Here's an example of how the perception system analyzes platforms:

```python
def _analyze_platforms(self, visible_platforms):
    """Analyze platforms by combining adjacent blocks into continuous platforms."""
    continuous_platforms = []
    current_platform = None
    
    for platform_info in sorted_platforms:
        if current_platform is None:
            current_platform = platform_info.copy()
        else:
            # Check if this block is adjacent (within 1 pixel)
            if platform_info['rect'].left - current_platform['rect'].right <= 1:
                # Extend current platform
                current_platform['rect'].width = (
                    platform_info['rect'].right - current_platform['rect'].left
                )
            else:
                # Gap found, store current platform and start new one
                continuous_platforms.append(current_platform)
                current_platform = platform_info.copy()
```

### Physics System

Our physics implementation balances realism with learnable mechanics. Key components include:

#### Movement Physics

```python
def accelerate(self):
    """Handles acceleration and friction for smooth movement."""
    self.change_x += self.direction * self.acceleration
    if self.on_ground:
        self.change_x += self.change_x * self.friction
    self.change_x = max(-self.max_speed_x, min(self.change_x, self.max_speed_x))
```

#### Jump Mechanics

```python
def jump(self):
    """Implements jump physics with initial velocity and gravity."""
    if self.on_ground:
        self.change_y = self.jump_speed
        self.on_ground = False
        self.is_jumping = True
```

## Training Process Deep Dive

### Reward System Design

Our reward system is carefully balanced to encourage desired behaviors while maintaining learning stability:

```python
def get_reward(self):
    """Calculate reward based on the agent's actions and state."""
    reward = 0
    
    # Base movement reward
    if self.agent.change_x > 0:
        reward += 1  # Encourage forward progress
    
    # Jumping rewards/penalties
    if state[IS_JUMPING]:
        if state[GAP_JUMPABLE] > 0:
            reward += 5  # Reward meaningful jumps
        else:
            reward -= 2  # Penalize unnecessary jumps
    
    # Completion reward
    if pygame.sprite.spritecollide(self.agent, self.level.goal_list, False):
        reward += 400  # Significant reward for reaching goal
        
    return reward
```

### Training Parameters

Here are some key parameters that affect training performance:

```python
class DQNAgent:
    def __init__(self):
        # Learning parameters
        self.gamma = 0.99 # Future reward discount
        self.tau = 0.001 # Soft update rate
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.02 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Exploration decay rate
        
        # Network parameters
        self.learning_rate = 0.0003
        self.batch_size = 128
        self.update_frequency = 500
```

## Installation and Setup

### Prerequisites

Before installing, ensure you have:
1. Python 3.8 or higher
2. CUDA-capable GPU (recommended for training)
3. Git for version control

## Usage Guide

### Training the AI
Command Line Arguments Summary
--m: Enable manual control
--l <number>: Select level (1-8)
--t: Enable training mode
--r: Enable visualization during training
--episodes <number>: Set number of training episodes
--lm <path>: Load a previously trained model

Start a basic training session:
```bash
python game.py --t --l 1 --r --episodes 1000
```

### Monitoring Training Progress

During training, you'll see real-time statistics:
```
Episode 100/1000:
  Steps: 245
  Distance to Goal: 320
  Average Reward: 125.7
  Loss: 0.0042
  Current ε: 0.607
```

These metrics help you understand how well the agent is learning:
- Steps: Lower is better, indicates more efficient paths
- Distance to Goal: Should decrease over time
- Average Reward: Should increase over time
- Loss: Should generally decrease but may fluctuate
- Epsilon (ε): Determines exploration vs exploitation
