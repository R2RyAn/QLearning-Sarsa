import random
import gym
import numpy as np
import matplotlib.pyplot as plt

np.bool8 = np.bool_

env = gym.make('Blackjack-v1', natural=True, sab=False)

rewards_per_episode = []
q_table = {}

def process_state(state):
    if isinstance(state, dict):  
        return tuple(state["observation"]) if "observation" in state else tuple(state.values())
    return tuple(state)

def choose_action(state, exploration_rate):
    # Epsilon-greedy policy.
    if random.random() < exploration_rate:
        return env.action_space.sample()
    else:
        q_values = [q_table.get((state, a), 0.0) for a in range(env.action_space.n)]
        return np.argmax(q_values)

def sarsa_learning(learning_rate, discount, exploration_rate, exploration_rate_decay, min_exploration_rate, episodes, max_steps):
    for episode in range(episodes):
        total_reward = 0
        if episode == int(episodes * 0.5):
            print("50% done...")

        state_dict, _ = env.reset() 
        state = process_state(state_dict)
        action = choose_action(state, exploration_rate)

        for step in range(max_steps):
            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            next_state = process_state(next_state_dict)
            next_action = choose_action(next_state, exploration_rate)

            old_value = q_table.get((state, action), 0.0)
            next_value = q_table.get((next_state, next_action), 0.0)

            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount * next_value)
            q_table[(state, action)] = new_value

            total_reward += reward
            state, action = next_state, next_action

            if terminated or truncated:
                break

        rewards_per_episode.append(total_reward)
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_rate_decay)

def plot_rewards(rewards, learning_rate, discount, exploration_rate_decay):
    window_size = 10000
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid') if len(rewards) >= window_size else rewards
    
    plt.plot(smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SARSA Learning: Rewards per Episode (Blackjack-v1)')
    plt.text(0.5, 0.1, f'LR={learning_rate}\nDiscount={discount}\nEpsilon Decay={exploration_rate_decay}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='cyan', alpha=0.5))
    plt.show()

if __name__ == '__main__':
    learning_rate = 0.1
    discount = 0.95
    exploration_rate = 1.0
    exploration_rate_decay = 0.9999
    min_exploration_rate = 0.01
    episodes = 500000
    max_steps = 100

    sarsa_learning(learning_rate, discount, exploration_rate, exploration_rate_decay, min_exploration_rate, episodes, max_steps)
    
    plot_rewards(rewards_per_episode, learning_rate, discount, exploration_rate_decay)
    
    env.close()
