import torch
from dqn_agent import DQNAgent  # Adjust the import based on your file structure
from gym_env import RunnerEnv  # Import your environment

# Set the state and action dimensions
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the environment
env = RunnerEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize the agent
agent = DQNAgent(state_dim, action_dim, device)

# Load the pre-trained weights
agent.load('dqn_agent.pth')

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
        # Example of choosing action with epsilon = 0 during evaluation
        action = agent.choose_action(state)

        # Step the environment
        next_state, reward, done, _ = env.step(action)

        # Update total reward
        total_reward += reward

        # Render the environment (optional)
        # env.render()

        # Move to the next state
        state = next_state

        q_values = agent.q_network(torch.tensor(state, dtype=torch.float32).to(device))  # Assuming state is a NumPy array
        # print(q_values)

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Close the environment when done
env.close()
