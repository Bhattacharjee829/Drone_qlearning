# Why Tabular Q-Learning Fails Under Uncertainty  
**Version 1 — Foundations and Baselines**

## Project Overview
Many drone navigation studies apply tabular Q-learning in uncertain environments such as dynamic congestion or changing conditions. While these methods often work in small, controlled settings, their limitations under uncertainty are rarely analyzed in detail.

This project studies **why tabular Q-learning fails under uncertainty**, and later uses Deep Q-Networks (DQN) as a diagnostic tool to explain this failure.  
Version 1 establishes the **formal foundations and baseline methods** required for this analysis.

---

## Version 1 Scope
Version 1 focuses on:
- A clear Markov Decision Process (MDP) formulation
- A small grid-based drone navigation environment
- Dynamic Programming (DP) baselines
- A tabular Q-learning baseline
- Clean experimental structure

---

## Environment Description
The environment is a small grid-based drone navigation task:
- Grid size: 5 × 5
- Start position: top-left corner
- Goal position: bottom-right corner
- One dynamically moving congestion cell
- Episode ends when the goal is reached or a step limit is exceeded

The environment provides:
- `reset()`
- `step(action)` → `(next_state, reward, done)`

---

## MDP Formulation
The task is modeled as a finite Markov Decision Process (MDP):

- **State**:  
  `(x, y, c)`  
  where `(x, y)` is the drone’s grid position and `c ∈ {0,1}` indicates whether the drone is currently in congestion.

- **Action Space**:  
  `{UP, DOWN, LEFT, RIGHT, STAY}`

- **Reward Function**:
  - +10 for reaching the goal
  - −10 for entering congestion or failure
  - −1 per time step

- **Transition Dynamics**:  
  The drone moves deterministically given an action.  
  Congestion moves stochastically, introducing uncertainty into the state transitions.

---

## Implemented Agents (Version 1)

### Dynamic Programming
- Value Iteration
- Policy Iteration (policy evaluation + policy improvement)

These methods operate on the explicit MDP model and serve as exact baselines on the small state space.

### Learning-Based Method
- Tabular Q-learning with:
  - Q-table
  - ε-greedy exploration
  - Standard Q-learning update rule

---

## Repository Structure
├── envs/
│ └── drone_grid_env.py
├── mdp/
│ ├── states.py
│ ├── actions.py
│ └── rewards.py
├── agents/
│ ├── value_iteration.py
│ ├── policy_iteration.py
│ └── tabular_qlearning.py
├── experiments/
│ └── run_tabular_baseline.py
├── README.md
