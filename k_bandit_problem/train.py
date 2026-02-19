import numpy as np
import random
import matplotlib.pyplot as plt
import logging

# configure basic logging for experiment monitoring
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
# many libraries (like matplotlib) emit verbose DEBUG logs; silence them
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# -----------------------------
# Environment
# -----------------------------
class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.random.normal(0, 1, k)
        self.optimal_action = np.argmax(self.q_true)

    def step(self, action):
        return np.random.normal(self.q_true[action], 1)


# -----------------------------
# Epsilon-Greedy (Sample Avg)
# -----------------------------
class EpsilonGreedy:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def select_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.k - 1)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])


# -----------------------------
# Epsilon-Greedy (Constant Step)
# -----------------------------
class EpsilonGreedyConstant:
    def __init__(self, k=10, epsilon=0.1, step_size=0.1):
        self.k = k
        self.epsilon = epsilon
        self.step_size = step_size
        self.Q = np.zeros(k)

    def select_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.k - 1)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.Q[action] += self.step_size * (reward - self.Q[action])


# -----------------------------
# UCB
# -----------------------------
class UCB:
    def __init__(self, k=10, c=2):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_action(self):
        self.t += 1

        for a in range(self.k):
            if self.N[a] == 0:
                return a

        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])

# -----------------------------
# Gradient Bandit
# -----------------------------
class GradientBandit:
    def __init__(self, k=10, alpha=0.1):
        self.k = k
        self.alpha = alpha
        self.H = np.zeros(k)
        self.avg_reward = 0
        self.t = 0

    def select_action(self):
        exp_H = np.exp(self.H - np.max(self.H))
        self.probs = exp_H / np.sum(exp_H)
        return np.random.choice(self.k, p=self.probs)

    def update(self, action, reward):
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t

        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - self.avg_reward) * (1 - self.probs[a])
            else:
                self.H[a] -= self.alpha * (reward - self.avg_reward) * self.probs[a]

# -----------------------------
# Optimistic Initial Values (Sample Avg)
# -----------------------------
class OptimisticInitialValues:
    def __init__(self, k=10, initial_value=5, alpha=0.1):
        self.k = k
        self.Q = np.full(k, float(initial_value), dtype=float)
        self.N = np.zeros(k, dtype=float)
        self.alpha = alpha
        # self.t = 0

    def select_action(self):
        # if self.t < self.k:
        #     return self.t

        max_value = np.max(self.Q)
        candidates = np.where(self.Q == max_value)[0]
        return np.random.choice(candidates)

    def update(self, action, reward):
        # self.t += 1
        self.N[action] += 1
        self.Q[action] += self.alpha * (reward - self.Q[action])

# -----------------------------
# Experiment
# -----------------------------
def run_experiment(agent_class, agent_kwargs, runs=2000, steps=1000):
    avg_rewards = np.zeros(steps)
    optimal_action_pct = np.zeros(steps)

    logging.info(f"Starting experiment: {agent_class.__name__} with {runs} runs, {steps} steps")

    for run in range(runs):
        if run % max(1, runs // 10) == 0:
            logging.debug(f"Beginning run {run + 1}/{runs}")

        bandit = Bandit()
        agent = agent_class(**agent_kwargs)

        for t in range(steps):
            action = agent.select_action()
            reward = bandit.step(action)
            agent.update(action, reward)

            # accumulate statistics
            avg_rewards[t] += reward

            if action == bandit.optimal_action:
                optimal_action_pct[t] += 1

            # log occasionally for debugging long runs
            if (t + 1) % max(1, steps // 5) == 0 and run == 0:
                # include the current estimate for the chosen action and a slice of Q for context
                q_val = getattr(agent, 'Q', None)
                q_str = ''
                if q_val is not None:
                    q_str = ' Q=' + ','.join(f"{v:.2f}" for v in q_val[:min(5, len(q_val))])
                    if len(q_val) > 5:
                        q_str += '...'
                logging.debug(f"run 1 step {t + 1}: action={action}, reward={reward:.2f}, optimal={action == bandit.optimal_action}{q_str}")

    avg_rewards /= runs
    optimal_action_pct = 100 * optimal_action_pct / runs

    # logging.info(f"Experiment completed for {agent_class.__name__}")
    return avg_rewards, optimal_action_pct


# -----------------------------
# Compare Methods
# -----------------------------
runs = 2000
steps = 1000

# eps_rewards, eps_optimal = run_experiment(EpsilonGreedy, {"epsilon": 0.1}, runs, steps)
# ucb_rewards, ucb_optimal = run_experiment(UCB, {"c": 2}, runs, steps)
# eps_const_rewards, eps_const_optimal = run_experiment(EpsilonGreedyConstant, {"epsilon": 0.1, "step_size": 0.1}, runs, steps)
# gradient_rewards, gradient_optimal = run_experiment(GradientBandit, {"alpha": 0.1}, runs, steps)
optimistic_rewards, optimistic_optimal = run_experiment(OptimisticInitialValues, {"initial_value": 5, "alpha": 0.1}, runs, steps)

def plot_results():
    # ---- Average Reward Plot ----
    plt.figure(figsize=(10,5))
    # plt.plot(eps_rewards, label="Epsilon-Greedy (ε=0.1)")
    # plt.plot(ucb_rewards, label="UCB (c=2)")
    # plt.plot(eps_const_rewards, label="Epsilon-Greedy-Const (ε=0.1, α=0.1)")
    # plt.plot(gradient_rewards, label="Gradient Bandit (α=0.1)")
    plt.plot(optimistic_rewards, label="Optimistic Initial Values (Q0=5)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Steps")
    plt.legend()
    plt.grid()
    plt.show()


    # ---- % Optimal Action Plot ----
    plt.figure(figsize=(10,5))
    # plt.plot(eps_optimal, label="Epsilon-Greedy (ε=0.1)")
    # plt.plot(ucb_optimal, label="UCB (c=2)")
    # plt.plot(eps_const_optimal, label="Epsilon-Greedy-Const (ε=0.1, α=0.1)")
    # plt.plot(gradient_optimal, label="Gradient Bandit (α=0.1)")
    plt.plot(optimistic_optimal, label="Optimistic Initial Values (Q0=5)")
    plt.xlabel("Steps")
    plt.ylabel("Optimal Action")
    plt.title("Optimal Action vs Steps")
    plt.legend()
    plt.grid()
    plt.show()

plot_results()
print("Finished experiments.")
