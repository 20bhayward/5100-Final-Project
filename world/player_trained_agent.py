import torch
"""
This script demonstrates how to use a pre-trained DQN agent to interact with a custom environment.

Modules:
    torch: PyTorch library for tensor computations and deep learning.
    dqn_agent: Custom module containing the DQNAgent class.
    gym_env: Custom module containing the RunnerEnv class.

Constants:
    state_dim (int): Dimension of the state space.
    action_dim (int): Dimension of the action space.
    device (torch.device): Device to run the computations on (CPU or GPU).

Classes:
    RunnerEnv: Custom environment class.
    DQNAgent: Custom DQN agent class.

Functions:
    None

Execution:
    1. Initialize the environment and the agent.
    2. Load the pre-trained weights into the agent.
    3. Set the agent to evaluation mode.
    4. Run a specified number of episodes where the agent interacts with the environment.
    5. For each episode, reset the environment and run until the episode is done.
    6. Choose actions based on the current state using the agent.
    7. Step the environment and accumulate rewards.
    8. Optionally render the environment.
    9. Print the total reward for each episode.
    10. Close the environment when done.
"""
from dqn_agent import DQNAgent  # Adjust the import based on your file structure
from gym_env import RunnerEnv  # Import your environment

# Set the state and action dimensions
state_dim = 7  # Replace with your actual state dimension
action_dim = 5  # Replace with your actual action dimension
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment
env = RunnerEnv()

# Initialize the agent
agent = DQNAgent(state_dim, action_dim, device)

# Load the pre-trained weights
agent.load('dqn_agent_2.pth')

# Set the agent to evaluation mode
agent.q_network.eval()  # For the QNetwork used during training
agent.target_network.eval()  # If using a target network

# Interact with the environment
num_episodes = 10  # Example: run 10 episodes
for episode in range(num_episodes):
    state = env.reset()  # Reset the environment
    done = False
    total_reward = 0

    while not done:
        # Choose an action based on the current state
        action = agent.choose_action(state, epsilon=0)

        # Step the environment
        next_state, reward, done, _ = env.step(action)

        # Update total reward
        total_reward += reward

        # Render the environment (optional)
        env.render()

        # Move to the next state
        state = next_state

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Close the environment when done
env.close()
