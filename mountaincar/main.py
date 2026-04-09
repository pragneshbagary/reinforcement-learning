import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from tilecoder import TileCoder
from semigradientsarsa import SemiGradientSARSA

env = gym.make('MountainCar-v0', render_mode="human")

coder = TileCoder(
    n_tilings=8, n_tiles=8,
    state_low=env.observation_space.low, # type: ignore
    state_high=env.observation_space.high # type: ignore
)

agent = SemiGradientSARSA(coder, n_actions=3, alpha=0.5, gamma=1.0, epsilon=0.1)

episode_lengths = []

for ep in range(500):
    state, _ = env.reset()
    action = agent.select_action(state)
    
    steps = 0
    total_reward = 0
    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        total_reward += reward
        done = terminated or truncated
        
        next_action = agent.select_action(next_state) if not done else 0
        agent.update(state, action, reward, next_state, next_action, done)
        
        state = next_state
        action = next_action
        steps += 1
        
        if done:
            break
    
    episode_lengths.append(steps)
    
    if (ep + 1) % 10 == 0:
        avg = np.mean(episode_lengths[-50:])
        print(f"Episode {ep+1}, Steps: {steps}, Total Reward: {total_reward}, Avg steps (last 50): {avg:.0f}")

env.close()

# Plot learning curve
plt.figure(figsize=(10, 4))
plt.plot(episode_lengths)
plt.xlabel('Episode')
plt.ylabel('Steps to reach goal')
plt.title('Semi-Gradient SARSA on MountainCar')
plt.show()



