import torch
import numpy as np
from dqn_agent import DQNAgent
from gym_env import RunnerEnv

def evaluate_trained_agent(env, agent, num_episodes=10):
    agent.q_network.eval()
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 1000:
            # Use zero epsilon for deterministic behavior
            action = agent.choose_action(state, epsilon=0.0)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            env.render()
            state = next_state
            steps += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    print(f"\nAverage Reward: {np.mean(total_rewards):.2f}")
    return np.mean(total_rewards)

# Initialize the environment
env = RunnerEnv()

# Set the state and action dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the agent and load trained weights
agent = DQNAgent(state_dim, action_dim, device)
agent.load('best_agent.pth')  # Path to the trained model weights

# Set the agent to evaluation mode
agent.q_network.eval()
agent.target_network.eval()

# Evaluate the trained agent
evaluate_trained_agent(env, agent, num_episodes=10)
