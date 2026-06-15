import numpy as np
from neural_network import NeuralNet
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=0.0005, gamma=1.0,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=500,
                 batch_size=64, target_update_freq=5):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.episode = 0

        self.state_mean = np.array([-0.3, 0.0])
        self.state_std = np.array([0.9, 0.035])

        self.q_net = NeuralNet(state_dim, 128, n_actions, lr=lr)
        self.target_net = NeuralNet(state_dim, 128, n_actions, lr=lr)
        self.sync_target()

        self.buffer = ReplayBuffer(capacity=50000)  # bigger buffer

        # Track best performance and save weights
        self.best_avg = 200
        self.best_weights = None

    def normalize(self, state):
        return (state - self.state_mean) / self.state_std

    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.episode / self.epsilon_decay)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_net.forward(self.normalize(state))
        return np.argmax(q)

    def select_action_greedy(self, state):
        q = self.q_net.forward(self.normalize(state))
        return np.argmax(q)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_norm = (states - self.state_mean) / self.state_std
        next_states_norm = (next_states - self.state_mean) / self.state_std

        q_all = self.q_net.forward(states_norm)
        q_current = q_all[np.arange(self.batch_size), actions]

        q_next_all = self.target_net.forward(next_states_norm)
        q_next_max = np.max(q_next_all, axis=1)
        targets = rewards + self.gamma * q_next_max * (1 - dones)

        deltas = targets - q_current
        dL_dz2 = np.zeros_like(q_all)
        dL_dz2[np.arange(self.batch_size), actions] = -2.0 * deltas

        self.q_net.forward(states_norm)
        self.q_net.backward(dL_dz2)

    def sync_target(self):
        self.target_net.W1 = self.q_net.W1.copy()
        self.target_net.b1 = self.q_net.b1.copy()
        self.target_net.W2 = self.q_net.W2.copy()
        self.target_net.b2 = self.q_net.b2.copy()

    def save_best(self, avg):
        if avg < self.best_avg:
            self.best_avg = avg
            self.best_weights = {
                'W1': self.q_net.W1.copy(), 'b1': self.q_net.b1.copy(),
                'W2': self.q_net.W2.copy(), 'b2': self.q_net.b2.copy()
            }

    def load_best(self):
        if self.best_weights:
            self.q_net.W1 = self.best_weights['W1'].copy()
            self.q_net.b1 = self.best_weights['b1'].copy()
            self.q_net.W2 = self.best_weights['W2'].copy()
            self.q_net.b2 = self.best_weights['b2'].copy()