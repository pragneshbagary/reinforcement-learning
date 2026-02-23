# Dynamic Programming — GridWorld

An implementation of the classic GridWorld problem from **Sutton & Barto's Reinforcement Learning: An Introduction (Chapter 4)**, built from scratch to understand Dynamic Programming.

---

## The Problem

A 4×4 grid where an agent must navigate to one of two terminal states (top-left or bottom-right). Every step costs -1 reward, encouraging the agent to find the shortest path.

```
 0    1    2    3
 4    5    6    7
 8    9   10   11
12   13   14   15
```

- **Terminal states:** 0 (top-left) and 15 (bottom-right)
- **Actions:** up, down, left, right
- **Reward:** -1 per step, 0 at terminal states
- **Wall rule:** hitting a wall keeps you in the same state

---

## Algorithms Implemented

### 1. Policy Evaluation
Computes how good a **uniform random policy** is by iteratively applying the Bellman equation until convergence:

```
V(s) = (1/4) * Σ_a [ r + γ V(s') ]
```

**Result — State values under random policy:**
```
  0.00 -13.00 -19.00 -21.00
-13.00 -17.00 -19.00 -19.00
-19.00 -19.00 -17.00 -13.00
-21.00 -19.00 -13.00   0.00
```
States farther from terminals have more negative values — under a random policy it takes on average 21 steps from the corners to escape.

---

### 2. Policy Improvement
Given the value function, extract the **greedy optimal policy**:

```
π'(s) = argmax_a [ r + γ V(s') ]
```

**Result — Optimal policy:**
```
T  ←  ←  ←
↑  ←  ←  ↓
↑  ↑  ↓  ↓
↑  →  →  T
```
Every arrow points toward a terminal state. T = terminal.

---

### 3. Value Iteration
Combines evaluation and improvement into a single update — finds optimal values directly without a fixed policy:

```
V(s) ← max_a [ r + γ V(s') ]
```

**Result — Optimal state values:**
```
  0.00   0.00  -1.00  -2.00
  0.00  -1.00  -2.00  -1.00
 -1.00  -2.00  -1.00   0.00
 -2.00  -1.00   0.00   0.00
```
Each value now represents the **minimum steps** needed to reach a terminal — much cleaner than the random policy values.

---

## Key Concepts

| Concept | What it answers |
|---|---|
| **Value Function V(s)** | How good is it to be in state s? |
| **Bellman Equation** | V(s) = immediate reward + discounted future value |
| **Policy Evaluation** | How good is this policy? |
| **Policy Improvement** | What's the best action given these values? |
| **Value Iteration** | Find optimal values directly |

---

## How to Run

```bash
python3 gridworld.py
```