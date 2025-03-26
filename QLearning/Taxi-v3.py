import random
import gym
import numpy as np
import matplotlib.pyplot as plt
np.bool8 = np.bool_

env = gym.make('Taxi-v3')
rewards_per_episode = []
q_table = np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(state, exploration_rate):
    if random.uniform(0,1)<exploration_rate:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

def learning_phase(learning_rate, discount, exploration_rate, exploration_rate_decay, min_exploration_rate, episodes, max_steps):
    for episode in range(episodes):
        total_reward = 0
        state, _ = env.reset()

        done = False

        for step in range(max_steps):
            action = choose_action(state, exploration_rate)

            next_state, reward, done, truncated, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])

            q_table[state, action] = (1 - learning_rate) * old_value + learning_rate * (reward + discount * next_max)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        rewards_per_episode.append(total_reward)
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_rate_decay)




def test_agent(max_steps):
    env = gym.make('Taxi-v3', render_mode='human')

    for episode in range(5):
        state, _ = env.reset()
        done = False

        print('Episode:', episode)
        for step in range(max_steps):
            env.render()
            action = np.argmax(q_table[state, :])
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state

            if done or truncated:
                env.render()
                print(f'Finished episode: {episode}, with reward: {reward}')
                break

env.close()



def plot_rewards_per_episode(rewards_per_episode, learning_rate, discount, exploration_rate_decay):
    window = 100
    if len(rewards_per_episode) >= window:
        moving_avg = np.convolve(rewards_per_episode, np.ones(window) / window, mode='valid')
        plt.plot(moving_avg, label=f'LR={learning_rate}, Gamma={discount}, Epsilon Decay={exploration_rate_decay}')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (100 episodes)')
        plt.title('Smoothed Reward per Episode (Taxi-v3)')

        # Add parameter info inside the plot
        plt.text(5000, -300,
                 f'Learning Rate={learning_rate}\nDiscount={discount}\nExploration Rate Decay={exploration_rate_decay}',
                 bbox=dict(facecolor='cyan', alpha=0.5))

        plt.show()

    else:
        print("Not enough episodes to compute moving average")


if __name__ == '__main__':
    learning_rate = 0.75
    discount = 0.97
    exploration_rate = 1
    exploration_rate_decay = 0.99
    min_exploration_rate = 0.01
    episodes = 10000
    max_steps = 100
    learning_phase(learning_rate, discount, exploration_rate, exploration_rate_decay, min_exploration_rate, episodes, max_steps)
    test_agent(max_steps)
    plot_rewards_per_episode(rewards_per_episode, learning_rate, discount, exploration_rate_decay)