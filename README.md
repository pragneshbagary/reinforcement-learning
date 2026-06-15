# Reinforcement Learning — Explorations

I've been diving into reinforcement learning from the ground up. This repo is where I work through ideas, implement algorithms from scratch, and build intuition for how agents actually learn to make decisions.

Most of the implementations are grounded in Sutton & Barto's *Reinforcement Learning: An Introduction* — not as an exercise, but because it's the clearest path from first principles to the kind of RL that matters in practice.

## What's here

| Directory | What I was trying to understand |
|-----------|----------------------------------|
| **k_bandit_problem** | The exploration-exploitation tradeoff — how do you balance trying new things vs. sticking with what works? |
| **dynamic_programming** | What does "optimal" even mean, and how do you compute it when you have a full model of the environment? |
| **mountaincar** | How do you scale RL to continuous state spaces without a lookup table? |
| **tic-tac-toe** | Can an agent learn a game purely through self-play with no prior knowledge of strategy? |

## Structure

```
├── k_bandit_problem/     # Exploration strategies on a k-armed bandit
├── dynamic_programming/  # Policy evaluation and value iteration in GridWorld
├── mountaincar/          # Semi-gradient SARSA with tile coding
└── tic-tac-toe/          # TD(0) agent, self-play training, playable GUI
```

## Running things

Each directory is self-contained, no dependencies beyond Python 3.

```bash
python k_bandit_problem/train.py
python dynamic_programming/gridworld.py
python mountaincar/main.py
python tic-tac-toe/train.py

# Play against the trained tic-tac-toe agent
python tic-tac-toe/play.py       # CLI
python tic-tac-toe/play_gui.py   # GUI
```

## Reference

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
