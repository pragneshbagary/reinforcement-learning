# MountainCar: Function Approximation with Semi-Gradient SARSA

This example demonstrates solving the MountainCar control problem using function approximation techniques from Chapter 9 of Sutton & Barto's "Reinforcement Learning: An Introduction".

## The MountainCar Problem

MountainCar is a classic reinforcement learning benchmark with a continuous state space. The car must learn to drive up a steep hill to reach the goal at the top. However, the car's engine is not powerful enough to climb the hill directly from the starting position. The optimal solution requires the car to drive back and forth to build momentum, creating a "swing" motion.

### State Space
- **Position**: Continuous value between -1.2 and 0.6
- **Velocity**: Continuous value between -0.07 and 0.07

### Action Space
- **0**: Push left (accelerate left)
- **1**: No push (coast)
- **2**: Push right (accelerate right)

### Reward
- **-1** for every time step until the goal is reached
- **0** when the goal (position ≥ 0.5) is achieved

## Approach: Semi-Gradient SARSA with Tile Coding

### Function Approximation
Since the state space is continuous, we cannot use tabular methods. Instead, we use **tile coding** to discretize the continuous state space into overlapping regions, creating a binary feature vector.

### Tile Coding Implementation
- **8 tilings** with **8×8 tiles** each
- Total of 512 features (8 × 64)
- Each tiling is offset to provide better coverage of the state space
- Features are binary (0 or 1) indicating which tile the current state falls into

### Semi-Gradient SARSA Algorithm
We implement the semi-gradient version of SARSA with separate weight vectors for each action:

- **Q-function**: Q(s,a) ≈ w_a · x(s), where x(s) is the tile-coded feature vector
- **Update rule**: w_a ← w_a + α·δ·x(s), where δ is the TD error
- **Learning rate**: α = 0.5 / n_tilings (normalized by number of tilings)
- **Discount factor**: γ = 1.0
- **Exploration**: ε-greedy with ε = 0.1

### Key Implementation Details

1. **Separate Weight Vectors**: One weight vector per action (3 total for MountainCar)
2. **Tiling Normalization**: Learning rate divided by number of tilings for stability
3. **Feature Encoding**: Each state is encoded into a 512-dimensional binary vector
4. **TD Target**: Uses bootstrapping with next state's Q-value estimate

## Files

- **`main.py`**: Training script that runs 500 episodes with visualization
- **`semigradientsarsa.py`**: Implementation of the Semi-Gradient SARSA agent
- **`tilecoder.py`**: Tile coding feature extractor for continuous states

## Results

The agent typically learns to solve MountainCar within 200-300 episodes, achieving the goal in under 200 steps per episode. The learning curve shows the characteristic "swing" behavior emerging as the agent discovers the optimal policy.

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
- Chapter 9: On-policy Prediction with Approximation
- Chapter 10: On-policy Control with Approximation