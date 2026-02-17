import random

class Agent():
    def __init__(self, epsilon = 0.1, alpha = 0.1):
        self.V = {}
        self.epsilon = epsilon
        self.alpha = alpha

    def get_value(self, state):
        if state not in self.V:
            self.V[state] = 0.5
        return self.V[state]

    def choose_action(self, state, legal_moves, board):
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        values = []

        for move in legal_moves:
            next_state = board.make_a_move(state, move, 'X')
            value = self.get_value(next_state)
            values.append((move,value))
        
        max_value = max(v for _,v in values)

        best_moves = [move for (move, v) in values if v == max_value]

        return random.choice(best_moves)

    def update(self, episode_states, final_reward):
        # Update backwards

        target = final_reward

        for state in reversed(episode_states):
            value = self.get_value(state)
            self.V[state] = value + self.alpha * (target - value)

            target = self.V[state]