class GridWorld:
    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.terminal_states = {0, 15}
        self.actions = ['up', 'down', 'left', 'right']

    def get_next_state(self, state, action):

        if state in self.terminal_states:
            return state

        row = state // self.cols
        col = state % self.cols

        if action == 'up':    row -= 1
        if action == 'down':  row += 1
        if action == 'left':  col -= 1
        if action == 'right': col += 1

        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return state

        return row * self.cols + col

    def get_reward(self, state):
        return 0 if state in self.terminal_states else -1
    
    def print_grid(self, values=None):
        for s in range(16):
            if values:
                print(f"{values[s]:6.2f}", end=' ')
            else:
                print(f"{s:4}", end=' ')
            if (s + 1) % self.cols == 0:
                print()

    def policy_evaluation(self, theta=0.0001, gamma=1.0):
        """
            V(s) = Σ_a π(a|s) Σ_s' p(s'|s,a) [ r + γ V(s') ]
            γ = 1
            Σ_s' p(s'|s,a) = 1
            V(s) = Σ_a π(a|s) [ r + V(s') ]
            π(a|s) = 1/4 for all a
            V(s) = (1/4) Σ_a [ r + V(s') ]
        """
        V = [0] * 16
        while True:
            delta = 0
            for s in range(16):
                if s in self.terminal_states:
                    continue
                v = V[s]
                new_v = 0
                for a in self.actions:
                    next_state = self.get_next_state(s, a)
                    reward = self.get_reward(next_state)
                    new_v += (1/4) * (reward + gamma * V[next_state])
                V[s] = new_v
                delta = max(delta, abs(v - new_v))
            if delta < theta:
                break
        return V
    

    def policy_improvement(self, V, gamma=1.0):
        """
            π'(s) = argmax_a [ r + γ V(s') ]
        """
        
        policy = {}
        for s in range(16):
            if s in self.terminal_states:
                policy[s] = None
                continue
            action_values = {}
            for a in self.actions:
                next_state = self.get_next_state(s, a)
                reward = self.get_reward(next_state)
                action_values[a] = reward + gamma * V[next_state]
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
        return policy
        
    def value_iteration(self, theta=0.0001, gamma=1.0):
        V = [0] * 16
        while True:
            delta = 0
            for s in range(16):
                if s in self.terminal_states:
                    continue
                v = V[s]
                action_values = []
                for a in self.actions:
                    next_state = self.get_next_state(s, a)
                    reward = self.get_reward(next_state)
                    action_values.append(reward + gamma * V[next_state])
                V[s] = max(action_values)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V


grid = GridWorld()
print("Initial Grid:")
grid.print_grid()
print("\nState Values under Uniform Random Policy:")
values = grid.policy_evaluation()
grid.print_grid(values)
print("\nOptimal Policy:")
optimal_policy = grid.policy_improvement(values)
arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→', None: 'T'}
for s in range(16):
    print(f"{arrow[optimal_policy[s]]:2}", end=' ')
    if (s + 1) % grid.cols == 0:
        print()

print("\nState Values under Optimal Policy:")
optimal_values = grid.value_iteration()
grid.print_grid(optimal_values)
