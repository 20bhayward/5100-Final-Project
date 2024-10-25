import torch
from dqn_agent import DQNAgent  # Adjust import if necessary
from gym_env import RunnerEnv  # Import the environment

# Set the state and action dimensions
state_dim = 6  # Adjust if different
action_dim = 3  # Adjust if different
device = torch.device("cpu")  # Use GPU if available

# Initialize the environment
env = RunnerEnv()

# Initialize the agent and load trained weights
agent = DQNAgent(state_dim, action_dim, device)
agent.load('dqn_agent_2.pth')  # Path to trained model weights

# Set the agent to evaluation mode
agent.q_network.eval()
agent.target_network.eval()

# Evaluation parameters
num_episodes = 10  # Number of test episodes
epsilon = 0.05  # Small epsilon for slight exploration

# Evaluation loop
total_rewards = []
for episode in range(num_episodes):
    state = env.reset()  # Reset environment at the start of each episode
    done = False
    episode_reward = 0

    while not done:
        # Choose action based on the current state with slight exploration
        action = agent.choose_action(state, epsilon=epsilon)

        # Step the environment with the chosen action
        next_state, reward, done, _ = env.step(action)

        # Accumulate rewards
        episode_reward += reward

        # Render the environment (optional)
        env.render()

        # Move to the next state
        state = next_state

    # Track total rewards per episode
    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

# Print summary of performance across episodes
average_reward = sum(total_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {average_reward}")

# Close the environment
env.close()
