import random
import gym
import numpy as np
np.bool8 = np.bool_

env = gym.make('CartPole-v1', render_mode='human')
env.reset()

episodes = 10

for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        _, reward, done, _, _ = env.step(action)
        score += reward
        env.render()
    print(f"Episode {episode} Score: {score}")

env.close()