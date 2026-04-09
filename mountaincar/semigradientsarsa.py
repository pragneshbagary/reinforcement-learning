import numpy as np

class SemiGradientSARSA:
    def __init__(self, coder, n_actions, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.coder = coder
        self.n_actions = n_actions
        self.w = np.zeros((n_actions, coder.n_features))
        self.alpha = alpha / coder.n_tilings
        self.gamma = gamma
        self.epsilon = epsilon
    
    def q_value(self, state, action):
        x = self.coder.encode(state)
        return self.w[action] @ x
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        q_values = [self.q_value(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)
        
    def select_action_greedy(self, state):
        q_values = [self.q_value(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, next_action, done):
        x = self.coder.encode(state)
        q_current = self.w[action] @ x
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_value(next_state, next_action)
        
        delta = target - q_current
        # The semi-gradient update: w ← w + α·δ·x(S,A)
        self.w[action] += self.alpha * delta * x