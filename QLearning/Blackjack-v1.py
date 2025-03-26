import random
import gym
import numpy as np
import matplotlib.pyplot as plt

# Fix for numpy bool issue
np.bool8 = np.bool_

# Create the Blackjack environment
env = gym.make('Blackjack-v1', natural=True, sab=False)

# Global storage
rewards_per_episode = []
q_table = {}

def process_state(state):
    """
    Convert the environment's returned state into a fully hashable tuple.
    - If 'state' is a dict containing 'observation', use that field.
    - Otherwise, if it's a dict but no 'observation' key, flatten the dict.
    - If it's already a tuple/list, turn it into a tuple of immutables.
    - If it's a primitive, just return it.
    """
    if isinstance(state, dict):
        # If it's the new Gym style: {'observation': (...), 'action_mask': ...}
        if "observation" in state:
            # Take the 'observation' field
            obs = state["observation"]
            return make_immutable(obs)
        else:
            # Flatten any dict into a tuple of its values (sorted by key just to be consistent)
            items = []
            for k in sorted(state.keys()):
                items.append(make_immutable(state[k]))
            return tuple(items)
    else:
        # It's not a dict, so just ensure it's an immutable tuple
        return make_immutable(state)

def make_immutable(obj):
    """
    Recursively convert lists/tuples/dicts to a fully immutable (hashable) structure (tuple).
    """
    if isinstance(obj, dict):
        # Flatten dict by sorted key order
        return tuple((k, make_immutable(obj[k])) for k in sorted(obj.keys()))
    elif isinstance(obj, (list, tuple)):
        return tuple(make_immutable(x) for x in obj)
    else:
        # int, float, bool, str, etc. are already hashable
        return obj

def learning_phase(learning_rate, discount, exploration_rate, exploration_rate_decay, min_exploration_rate, episodes, max_steps):
    for episode in range(episodes):
        total_reward = 0
        # Just printing progress
        if episode == int(episodes * 0.5):
            print("50% done...")

        # Initial state
        state_dict = env.reset()
        state = process_state(state_dict)

        for step in range(max_steps):
            action = choose_action(state, exploration_rate)
            next_state_dict, reward, terminated, truncated, _ = env.step(action)

            next_state = process_state(next_state_dict)

            # Q-learning update
            old_value = q_table.get((state, action), 0.0)
            next_max = max(q_table.get((next_state, a), 0.0) for a in range(env.action_space.n))
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount * next_max)
            q_table[(state, action)] = new_value

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards_per_episode.append(total_reward)

        # Decay exploration
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_rate_decay)

def choose_action(state, exploration_rate):
    # Epsilon-greedy choice
    if random.random() < exploration_rate:
        return env.action_space.sample()
    else:
        # Exploit: pick the action with max Q-value
        q_values = [q_table.get((state, a), 0.0) for a in range(env.action_space.n)]
        return np.argmax(q_values)

def test_agent(max_steps):
    test_env = gym.make('Blackjack-v1', render_mode='human')
    for episode in range(5):
        # Reset + process state
        state_dict = test_env.reset()
        state = process_state(state_dict)

        print("Episode:", episode)
        for step in range(max_steps):
            test_env.render()

            q_values = [q_table.get((state, a), 0.0) for a in range(test_env.action_space.n)]
            action = np.argmax(q_values)

            next_state_dict, reward, terminated, truncated, _ = test_env.step(action)
            next_state = process_state(next_state_dict)

            state = next_state

            if terminated or truncated:
                print(f"Reward in this episode: {reward}")
                break

    test_env.close()

def plot_rewards_per_episode(rewards, learning_rate, discount, exploration_rate_decay):
    # Optionally smooth the reward curve
    window_size = 10000
    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(rewards,
                                       np.ones(window_size) / window_size,
                                       mode='valid')
    else:
        smoothed_rewards = rewards  # not enough data to smooth

    plt.plot(smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode (Blackjack-v1)')

    # Use axes coordinates to position the text box
    # For example, x=0.7, y=0.75 means 70% of the way from the left,
    # and 75% of the way from the bottom, in the plot area.
    plt.text(
        0.5, 0.1,
        f'Learning Rate={learning_rate}\n'
        f'Discount={discount}\n'
        f'Exploration Rate Decay={exploration_rate_decay}',
        transform=plt.gca().transAxes,  # <--- critical to use axes coords
        bbox=dict(facecolor='cyan', alpha=0.5)
    )

    plt.show()

if __name__ == '__main__':
    # ✅ Each variable is assigned as a single float or int, no trailing commas.
    learning_rate = 0.1
    discount = 0.95
    exploration_rate = 1
    exploration_rate_decay = 0.9999
    min_exploration_rate = 0.01
    episodes = 500000
    max_steps = 100

    # Train the Q-learning agent
    learning_phase(
        learning_rate,
        discount,
        exploration_rate,
        exploration_rate_decay,
        min_exploration_rate,
        episodes,
        max_steps
    )

    # Plot training progress
    plot_rewards_per_episode(rewards_per_episode, learning_rate, discount, exploration_rate_decay)

    # Test the trained agent visually (5 episodes)
    # test_agent(max_steps=100)

    # Close the main env
    env.close()
