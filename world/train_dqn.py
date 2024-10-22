import torch
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
        # env.render()
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
    print(f"Trained model saved as {path_str}")

env.close()
