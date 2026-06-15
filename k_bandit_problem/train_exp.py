import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# Environment
# -----------------------------
class Bandit:
    """
    k-armed bandit with pluggable reward distributions.

    dist options:
      "gaussian"      — standard S&B testbed (baseline)
      "bernoulli"     — binary rewards, probability from sigmoid of q_true
      "cauchy"        — heavy-tailed, no finite variance (stress test for UCB)
      "uniform"       — uniform noise ± 1 around q_true

    nonstationary=True — q_true drifts by a random walk each step.
    drift_std controls the walk speed (S&B uses 0.01).
    """
    def __init__(self, k=10, dist="gaussian", nonstationary=False, drift_std=0.01):
        self.k = k
        self.dist = dist
        self.nonstationary = nonstationary
        self.drift_std = drift_std
        self.q_true = np.random.normal(0, 1, k)
        self.optimal_action = np.argmax(self.q_true)

    def step(self, action):
        if self.nonstationary:
            self.q_true += np.random.normal(0, self.drift_std, self.k)
            self.optimal_action = np.argmax(self.q_true)

        q = self.q_true[action]

        if self.dist == "gaussian":
            return np.random.normal(q, 1)
        elif self.dist == "bernoulli":
            p = 1 / (1 + np.exp(-q))           # sigmoid → valid probability
            return float(np.random.binomial(1, p))
        elif self.dist == "cauchy":
            return q + np.random.standard_cauchy()
        elif self.dist == "uniform":
            return np.random.uniform(q - 1, q + 1)
        else:
            raise ValueError(f"Unknown dist: {self.dist}")


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
# Optimistic Initial Values
# -----------------------------
class OptimisticInitialValues:
    def __init__(self, k=10, initial_value=5, alpha=0.1):
        self.k = k
        self.Q = np.full(k, float(initial_value), dtype=float)
        self.N = np.zeros(k, dtype=float)
        self.alpha = alpha

    def select_action(self):
        max_value = np.max(self.Q)
        candidates = np.where(self.Q == max_value)[0]
        return np.random.choice(candidates)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += self.alpha * (reward - self.Q[action])


# -----------------------------
# Experiment runner
# -----------------------------
def run_experiment(agent_class, agent_kwargs, bandit_kwargs=None, runs=2000, steps=1000):
    """
    bandit_kwargs — passed directly to Bandit().
    Leave as None for the default S&B gaussian testbed.
    """
    if bandit_kwargs is None:
        bandit_kwargs = {}

    avg_rewards = np.zeros(steps)
    optimal_action_pct = np.zeros(steps)

    label = f"{agent_class.__name__}({agent_kwargs})"
    dist_label = bandit_kwargs.get("dist", "gaussian")
    nonstat = bandit_kwargs.get("nonstationary", False)
    print(f"\nRunning: {label} | dist={dist_label} | nonstationary={nonstat}")

    for run in range(runs):
        if run % max(1, runs // 5) == 0:
            print(f"  run {run + 1}/{runs}")

        bandit = Bandit(**bandit_kwargs)
        agent = agent_class(k=bandit.k, **agent_kwargs)

        for t in range(steps):
            action = agent.select_action()
            reward = bandit.step(action)
            agent.update(action, reward)
            avg_rewards[t] += reward
            if action == bandit.optimal_action:
                optimal_action_pct[t] += 1

    avg_rewards /= runs
    optimal_action_pct = 100 * optimal_action_pct / runs
    return avg_rewards, optimal_action_pct


# -----------------------------
# Plot helper
# -----------------------------
def plot_results(results, title="10-Armed Bandit Testbed"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for label, rewards, optimal in results:
        ax1.plot(rewards, label=label, linewidth=1.5, alpha=0.85)
        ax2.plot(optimal, label=label, linewidth=1.5, alpha=0.85)

    ax1.set_ylabel("Average Reward")
    ax1.set_xlabel("Steps")
    ax1.set_title("Average Reward vs Steps")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_ylabel("% Optimal Action")
    ax2.set_xlabel("Steps")
    ax2.set_title("Optimal Action % vs Steps")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()
    print(f"Saved: {title.replace(' ', '_').lower()}.png")


# -----------------------------
# Agents to compare (shared)
# -----------------------------
AGENTS = [
    (EpsilonGreedy,        {"epsilon": 0.1},                  "ε-Greedy (ε=0.1)"),
    (UCB,                  {"c": 2},                          "UCB (c=2)"),
    (EpsilonGreedyConstant,{"epsilon": 0.1, "step_size": 0.1},"ε-Greedy-Const (α=0.1)"),
    (GradientBandit,       {"alpha": 0.1},                    "Gradient Bandit (α=0.1)"),
    (OptimisticInitialValues, {"initial_value": 5, "alpha": 0.1}, "Optimistic (Q0=5)"),
]

RUNS  = 2000
STEPS = 1000


# ================================================================
# EXPERIMENT 1 — Baseline (reproduce S&B Figure 2.6 exactly)
# ================================================================
print("=" * 60)
print("EXPERIMENT 1: Baseline — Gaussian rewards (S&B Figure 2.6)")
print("=" * 60)

baseline_results = []
for cls, kwargs, label in AGENTS:
    r, o = run_experiment(cls, kwargs, bandit_kwargs={"dist": "gaussian"}, runs=RUNS, steps=STEPS)
    baseline_results.append((label, r, o))

plot_results(baseline_results, title="Baseline — Gaussian Rewards (S&B Fig 2.6)")


# ================================================================
# EXPERIMENT 2 — Swap reward distributions (one agent at a time)
# ================================================================
# Pick one agent (UCB) and show how it responds to each distribution.
# This is your core "stress test" finding.
print("\n" + "=" * 60)
print("EXPERIMENT 2: UCB across reward distributions")
print("=" * 60)

DISTS = ["gaussian", "bernoulli", "cauchy", "uniform"]
ucb_dist_results = []
for dist in DISTS:
    r, o = run_experiment(UCB, {"c": 2}, bandit_kwargs={"dist": dist}, runs=RUNS, steps=STEPS)
    ucb_dist_results.append((f"UCB | dist={dist}", r, o))

plot_results(ucb_dist_results, title="UCB — Effect of Reward Distribution")


# ================================================================
# EXPERIMENT 3 — All agents under Cauchy (heavy-tailed stress test)
# ================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: All agents — Cauchy (heavy-tailed) rewards")
print("=" * 60)

cauchy_results = []
for cls, kwargs, label in AGENTS:
    r, o = run_experiment(cls, kwargs, bandit_kwargs={"dist": "cauchy"}, runs=RUNS, steps=STEPS)
    cauchy_results.append((label, r, o))

plot_results(cauchy_results, title="All Agents — Cauchy (Heavy-Tailed) Rewards")


# ================================================================
# EXPERIMENT 4 — Nonstationary rewards (random walk)
# ================================================================
# NOTE: sample-average agents (EpsilonGreedy, UCB) should degrade here.
# Constant step-size agents (EpsilonGreedyConstant) should hold up better.
# This is a known theoretical prediction — verify it empirically.
print("\n" + "=" * 60)
print("EXPERIMENT 4: All agents — Nonstationary rewards (random walk)")
print("=" * 60)

nonstat_results = []
for cls, kwargs, label in AGENTS:
    r, o = run_experiment(
        cls, kwargs,
        bandit_kwargs={"dist": "gaussian", "nonstationary": True, "drift_std": 0.01},
        runs=RUNS, steps=STEPS
    )
    nonstat_results.append((label, r, o))

plot_results(nonstat_results, title="All Agents — Nonstationary Rewards (Random Walk)")

print("\nAll experiments complete.")