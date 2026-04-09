# Reinforcement Learning: An Introduction - Implementation Examples

This repository contains Python implementations of reinforcement learning algorithms and examples from Sutton & Barto's "Reinforcement Learning: An Introduction (2nd ed.)".

## Overview

This repository implements key reinforcement learning concepts from Sutton & Barto's textbook:

- **Temporal Difference (TD) Learning** — Learning value functions from experience without knowing the environment model
- **Multi-Armed Bandits** — Exploration-exploitation tradeoff with various action-selection strategies
- **Dynamic Programming** — Computing optimal policies and value functions when the environment model is known
- **Function Approximation** — Using tile coding and semi-gradient methods for continuous state spaces

## Examples

| Example | Topic | Description |
|---------|-------|-------------|
| **tic-tac-toe** | Temporal Difference Learning | TD(0) agent that learns to play tic-tac-toe through self-play, with GUI for testing trained agents |
| **k_bandit_problem** | Multi-Armed Bandits | Comparison of action-selection strategies (epsilon-greedy, UCB, gradient bandit, optimistic initial values) on a k-armed bandit problem with performance metrics |
| **dynamic_programming** | Dynamic Programming | GridWorld implementation demonstrating policy evaluation, policy improvement, and value iteration algorithms |
| **mountaincar** | Function Approximation | Semi-gradient SARSA with tile coding for solving the continuous MountainCar control problem |

## Directory Structure

```
├── tic-tac-toe/          # TD learning for tic-tac-toe
│   ├── agent.py          # Agent with TD learning
│   ├── board.py          # Game environment
│   ├── train.py          # Training script
│   ├── play.py           # CLI gameplay
│   └── play_gui.py       # GUI for playing against trained agent
│
├── k_bandit_problem/     # Multi-armed bandit experiments
│   ├── train.py          # Bandit environment and agents
│   └── plots/            # Experiment results
│
├── dynamic_programming/  # GridWorld DP algorithms
│   └── gridworld.py      # Policy evaluation, improvement, and value iteration
│
└── mountaincar/          # Function approximation on MountainCar
    ├── main.py           # Training script with visualization
    ├── semigradientsarsa.py  # Semi-gradient SARSA agent
    └── tilecoder.py      # Tile coding implementation
```

## Requirements

- Python 3.x
- No external dependencies (each example is self-contained)

## Usage

Each example can be run from its directory:

```bash
# Train tic-tac-toe agent
cd tic-tac-toe
python train.py

# Run k-bandit experiments
cd k_bandit_problem
python train.py

# Run GridWorld DP algorithms
cd dynamic_programming
python gridworld.py

# Train MountainCar agent
cd mountaincar
python main.py
```

For tic-tac-toe, you can also play against the trained agent:

```bash
python play.py          # CLI version
python play_gui.py      # GUI version
```

## References

Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
