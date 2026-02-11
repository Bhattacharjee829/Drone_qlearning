# agents/value_iteration.py


from typing import Dict, Tuple
from mdp.actions  import ACTIONS, UP, DOWN, LEFT, RIGHT, STAY
from mdp.rewards import GOAL_REWARD, CONGESTION_PENALTY, STEP_COST


def _move(x: int, y: int, action: int, grid_size: int) -> Tuple[int, int]:
    """Deterministic movement on the grid with boundary clipping."""
    if action == UP:
        y = max(0, y - 1)
    elif action == DOWN:
        y = min(grid_size - 1, y + 1)
    elif action == LEFT:
        x = max(0, x - 1)
    elif action == RIGHT:
        x = min(grid_size - 1, x + 1)
    elif action == STAY:
        pass
    return x, y


def value_iteration(
    grid_size: int = 5,
    gamma: float = 0.95,
    theta: float = 1e-6,
    max_iters: int = 10_000,
    start: Tuple[int, int] = (0, 0),
    goal: Tuple[int, int] = (4, 4),
) -> Tuple[Dict[Tuple[int, int, int], float], Dict[Tuple[int, int, int], int]]:
   


    p_cong_next = 1.0 / (grid_size * grid_size)

    # Initialize V(s) = 0
    V: Dict[Tuple[int, int, int], float] = {}
    for x in range(grid_size):
        for y in range(grid_size):
            for c in (0, 1):
                V[(x, y, c)] = 0.0

    def expected_reward_and_next_value(x: int, y: int, a: int) -> float:
       
        nx, ny = _move(x, y, a, grid_size)

        # If you reach the goal, terminate (no future value).
        if (nx, ny) == goal:
            return STEP_COST + GOAL_REWARD

        # Otherwise expected reward includes expected congestion penalty
        r_exp = STEP_COST + (p_cong_next * CONGESTION_PENALTY)

        # Expected next value under random c'
        v_next = (1.0 - p_cong_next) * V[(nx, ny, 0)] + p_cong_next * V[(nx, ny, 1)]
        return r_exp + gamma * v_next

    # Value Iteration loop
    for _ in range(max_iters):
        delta = 0.0

        for x in range(grid_size):
            for y in range(grid_size):
                for c in (0, 1):
                    # Terminal goal states can be kept at 0 (or any constant)
                    if (x, y) == goal:
                        continue

                    old_v = V[(x, y, c)]

                    # Bellman optimality backup
                    best_q = float("-inf")
                    for a in ACTIONS:
                        q = expected_reward_and_next_value(x, y, a)
                        if q > best_q:
                            best_q = q

                    V[(x, y, c)] = best_q
                    delta = max(delta, abs(old_v - best_q))

        if delta < theta:
            break

    # Extract greedy policy
    pi: Dict[Tuple[int, int, int], int] = {}
    for x in range(grid_size):
        for y in range(grid_size):
            for c in (0, 1):
                if (x, y) == goal:
                    pi[(x, y, c)] = STAY
                    continue

                best_a = ACTIONS[0]
                best_q = float("-inf")
                for a in ACTIONS:
                    q = expected_reward_and_next_value(x, y, a)
                    if q > best_q:
                        best_q = q
                        best_a = a
                pi[(x, y, c)] = best_a

    return V, pi


# Quick test
if __name__ == "__main__":
    V, pi = value_iteration(grid_size=5, goal=(4, 4))
    print("V(start, no congestion) =", V[(0, 0, 0)])
    print("Best action at start =", pi[(0, 0, 0)])
