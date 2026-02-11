# agents/tabular_qlearning.py


import random
from typing import Dict, Tuple, List
from mdp.actions import ACTIONS
from mdp.states import State


class TabularQLearningAgent:
    def __init__(
        self,
        grid_size: int = 5,
        alpha: float = 0.1,   # learning rate
        gamma: float = 0.95,  # discount
        epsilon: float = 0.1  # exploration rate
    ):
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: maps state_index -> list of Q-values for each action
        self.Q: Dict[int, List[float]] = {}

    def _ensure_state(self, s_idx: int) -> None:
        """Create Q-values for unseen states."""
        if s_idx not in self.Q:
            self.Q[s_idx] = [0.0 for _ in range(len(ACTIONS))]

    def select_action(self, state: Tuple[int, int, int]) -> int:
        """
        ε-greedy policy:
          with prob ε choose random action
          else choose greedy action from Q-table
        """
        s = State(*state)
        s_idx = s.to_index(self.grid_size)
        self._ensure_state(s_idx)

        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        # greedy
        q_values = self.Q[s_idx]
        best_a = max(range(len(ACTIONS)), key=lambda a: q_values[a])
        return ACTIONS[best_a]

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        done: bool
    ) -> None:
        """
        Q-learning update:
          Q(s,a) <- Q(s,a) + alpha * [ r + gamma*max_a' Q(s',a') - Q(s,a) ]
        """
        s = State(*state)
        ns = State(*next_state)

        s_idx = s.to_index(self.grid_size)
        ns_idx = ns.to_index(self.grid_size)

        self._ensure_state(s_idx)
        self._ensure_state(ns_idx)

        a_idx = ACTIONS.index(action)

        q_sa = self.Q[s_idx][a_idx]
        target = reward

        if not done:
            target = reward + self.gamma * max(self.Q[ns_idx])

        self.Q[s_idx][a_idx] = q_sa + self.alpha * (target - q_sa)

    def train(self, env, episodes: int = 200) -> List[float]:
        """
        Simple training loop.
        Returns list of total rewards per episode.
        """
        returns = []

        for _ in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = env.step(action)

                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            returns.append(total_reward)

        return returns


# Quick test
if __name__ == "__main__":
    from envs.drone_grid_env import DroneGridEnv

    env = DroneGridEnv(grid_size=5, max_steps=50)
    agent = TabularQLearningAgent(grid_size=5, alpha=0.1, gamma=0.95, epsilon=0.1)

    rets = agent.train(env, episodes=200)
    print("Training finished.")
    print("Last 10 episode returns:", rets[-10:])
    print("Q-table size (#visited states):", len(agent.Q))
