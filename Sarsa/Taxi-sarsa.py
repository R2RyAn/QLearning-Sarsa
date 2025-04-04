import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
np.bool8 = np.bool_

env = gym.make('Taxi-v3')

def initialize_q_table(env):
    return np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])   

def sarsa_learning(q_table, learning_rate, discount, epsilon, epsilon_decay, min_epsilon, episodes, max_steps):
    rewards_per_episode = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        action = choose_action(state, q_table, epsilon)
        total_reward = 0
        done = False
        
        for step in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = choose_action(next_state, q_table, epsilon) if not done else None
            
            if not done:
                q_table[state, action] += learning_rate * (reward + discount * q_table[next_state, next_action] - q_table[state, action])
            else:
                q_table[state, action] += learning_rate * (reward - q_table[state, action])
                
            total_reward += reward
            state = next_state
            action = next_action
            
            if done or truncated:
                break
        
        rewards_per_episode.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
    return rewards_per_episode

def plot_learning_curve(rewards_sarsa, episodes):
    window = 100
    if len(rewards_sarsa) >= window:
        moving_avg_sarsa = np.convolve(rewards_sarsa, np.ones(window) / window, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg_sarsa, label='SARSA', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (100 episodes)')
        plt.title('Learning Curve: SARSA (Taxi-v3)')
        plt.legend()
        plt.grid(True)  
        plt.text(0.5 * len(moving_avg_sarsa), min(moving_avg_sarsa) + 1, 
                 f'LR={learning_rate}\nDiscount={discount}\nEpsilon Decay={epsilon_decay}',
                 bbox=dict(facecolor='cyan', alpha=0.5))
        plt.show()
    else:
        print("Not enough episodes to compute moving average")

if __name__ == '__main__':
    learning_rate = 0.7
    discount = 0.1
    epsilon = 0.999
    epsilon_decay = 0.99
    min_epsilon = 0.01
    episodes = 10000
    max_steps = 100

    q_table_sarsa = initialize_q_table(env)
    rewards_sarsa = sarsa_learning(q_table_sarsa, learning_rate, discount, epsilon, epsilon_decay, min_epsilon, episodes, max_steps)
    
    plot_learning_curve(rewards_sarsa, episodes)

    def multi_agent_sarsa(q_table, learning_rate, discount, epsilon, epsilon_decay, min_epsilon, episodes, max_steps, num_agents):
        rewards_per_episode = []
        
        for episode in range(episodes):
            total_reward = np.zeros(num_agents)
            states = [env.reset() for _ in range(num_agents)]  
            actions = [choose_action(state, q_table, epsilon) for state in states]
            done = [False] * num_agents
            
            for step in range(max_steps):
                next_states = []
                rewards = []
                next_actions = []
                
                for i in range(num_agents):
                    next_state, reward, done[i], truncated, _ = env.step(actions[i])
                    next_actions.append(choose_action(next_state, q_table, epsilon) if not done[i] else None)
                    next_states.append(next_state)
                    rewards.append(reward)
                    total_reward[i] += reward
                    states[i] = next_state
                    actions[i] = next_actions[i]

                for i in range(num_agents):
                    if not done[i]:
                        q_table[states[i], actions[i]] += learning_rate * (rewards[i] + discount * q_table[next_states[i], next_actions[i]] - q_table[states[i], actions[i]])
                    else:
                        q_table[states[i], actions[i]] += learning_rate * (rewards[i] - q_table[states[i], actions[i]])

                if all(done):  
                    break
            
            rewards_per_episode.append(np.sum(total_reward)) 
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        return rewards_per_episode
