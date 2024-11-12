import torch
"""
This script trains a Deep Q-Network (DQN) agent in a custom Runner environment using PyTorch.

Modules:
    torch: PyTorch library for tensor computations and neural networks.
    numpy: Library for numerical computations.
    dqn_agent: Custom module containing the DQNAgent class.
    replay_buffer: Custom module containing the ReplayBuffer class.
    gym_env: Custom module containing the RunnerEnv class.

Functions:
    None

Variables:
    device (torch.device): The device to run the computations on (MPS if available, otherwise CPU).
    env (RunnerEnv): The custom Runner environment.
    state_dim (int): The dimension of the state space.
    action_dim (int): The dimension of the action space.
    agent (DQNAgent): The DQN agent.
    replay_buffer (ReplayBuffer): The replay buffer for storing experience tuples.
    n_episodes (int): The number of episodes to train the agent.
    target_update_freq (int): The frequency (in episodes) to update the target network.
    batch_size (int): The batch size for training the agent.

Training Loop:
        state (np.ndarray): The initial state of the environment.
        done (bool): Flag indicating whether the episode is done.
        total_reward (float): The total reward accumulated in the episode.

            action (int): The action chosen by the agent.
            next_state (np.ndarray): The next state after taking the action.
            reward (float): The reward received after taking the action.
            done (bool): Flag indicating whether the episode is done.
            total_reward (float): The total reward accumulated in the episode.

            Store the experience tuple in the replay buffer and train the agent.

        Update the target network every `target_update_freq` episodes.
        Decay the agent's epsilon value.
        Print the total reward for the episode.
        Save the agent's model.

    Close the environment.
"""
import numpy as np
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from gym_env import RunnerEnv

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize environment
env = RunnerEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim, device)
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=64)

n_episodes = 10000
target_update_freq = 10
batch_size = 64

for episode in range(n_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        replay_buffer.store(state, action, reward, next_state, done)
        agent.train(replay_buffer, batch_size)

        state = next_state

    # Update target network every few episodes
    if episode % target_update_freq == 0:
        agent.update_target_network()

    agent.decay_epsilon()
    print(f"Episode {episode}, Total Reward: {total_reward}")

    path_str = "dqn_agent_2.pth"
    agent.save(path_str)

env.close()
