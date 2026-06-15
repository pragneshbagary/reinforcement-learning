import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

env = gym.make('MountainCar-v0', render_mode='human')
env.metadata['render_fps'] = 120 
agent = DQNAgent(state_dim=2, n_actions=3, lr=0.0005)
episode_lengths = []

# Fixed test states to monitor Q-values over time
test_states = [
    np.array([-0.5, 0.0]),    # bottom of valley, no velocity
    np.array([-0.5, 0.05]),   # bottom, moving right
    np.array([-0.5, -0.05]),  # bottom, moving left
    np.array([0.4, 0.03]),    # near goal, moving right
]
q_history = {i: [] for i in range(len(test_states))}

for ep in range(2000):
    state, _ = env.reset()
    agent.episode = ep
    steps = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Reward shaping: encourage gaining speed and height
        shaped_reward = reward + 300 * (abs(next_state[1]) - abs(state[1]))  # speed bonus
        shaped_reward += 100 * (next_state[0] - state[0]) * (1 if next_state[0] > -0.5 else -1)  # height on right side

        agent.buffer.push(state, action, shaped_reward, next_state, float(done))

        for _ in range(4):
            agent.update()

        state = next_state
        steps += 1

        if done:
            break

    if (ep + 1) % agent.target_update_freq == 0:
        agent.sync_target()

    episode_lengths.append(steps)

    if (ep + 1) % 50 == 0:
        avg = np.mean(episode_lengths[-50:])

        for i, s in enumerate(test_states):
            q = agent.q_net.forward(agent.normalize(s))
            q_history[i].append(q.copy())

        q_sample = agent.q_net.forward(agent.normalize(test_states[0]))
        q_spread = np.max(q_sample) - np.min(q_sample)
        w1_norm = np.linalg.norm(agent.q_net.W1)
        w2_norm = np.linalg.norm(agent.q_net.W2)
        actions_chosen = [np.argmax(agent.q_net.forward(agent.normalize(s))) for s in test_states]

        print(f"Ep {ep+1:4d} | ε={agent.epsilon:.3f} | Steps={avg:.0f} | "
              f"Q(bottom)=[{q_sample[0]:.1f}, {q_sample[1]:.1f}, {q_sample[2]:.1f}] | "
              f"Spread={q_spread:.2f} | "
              f"|W1|={w1_norm:.1f} |W2|={w2_norm:.1f} | "
              f"Actions={actions_chosen}")

env.close()