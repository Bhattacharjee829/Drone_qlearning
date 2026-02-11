# agents/policy_iteration.py


from typing import Dict, Tuple
from mdp.actions import ACTIONS, UP, DOWN, LEFT, RIGHT, STAY
from mdp.rewards import GOAL_REWARD, CONGESTION_PENALTY, STEP_COST


def _move(x: int, y: int, action: int, grid_size: int) -> Tuple[int, int]:
    """Deterministic movement with boundary clipping."""
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


def policy_iteration(
    grid_size: int = 5,
    gamma: float = 0.95,
    eval_theta: float = 1e-6,
    eval_max_iters: int = 10_000,
    improve_max_iters: int = 1_000,
    start: Tuple[int, int] = (0, 0),
    goal: Tuple[int, int] = (4, 4),
) -> Tuple[Dict[Tuple[int, int, int], float], Dict[Tuple[int, int, int], int]]:
    """
    Policy Iteration on the MDP:
      state = (x, y, c) where c in {0,1} indicates "currently congested".

    Congestion model:
      Each step, the congestion cell is uniformly random over the grid.
      So the probability the agent is congested after moving is:
        p = 1/(grid_size^2)

    Returns:
      V: dict mapping state -> value under the final policy
      pi: dict mapping state -> action (deterministic policy)
    """

    p_cong_next = 1.0 / (grid_size * grid_size)

    # Initialize value function
    V: Dict[Tuple[int, int, int], float] = {}
    for x in range(grid_size):
        for y in range(grid_size):
            for c in (0, 1):
                V[(x, y, c)] = 0.0

    # Initialize a simple policy: always RIGHT (unless at right edge, then DOWN)
    pi: Dict[Tuple[int, int, int], int] = {}
    for x in range(grid_size):
        for y in range(grid_size):
            for c in (0, 1):
                if (x, y) == goal:
                    pi[(x, y, c)] = STAY
                elif x < grid_size - 1:
                    pi[(x, y, c)] = RIGHT
                else:
                    pi[(x, y, c)] = DOWN

    def q_value_under_model(x: int, y: int, a: int) -> float:
        """
        Compute Q(s,a) using our simple expected model.

        - Move deterministically to (nx,ny)
        - If goal reached: terminate with STEP_COST + GOAL_REWARD
        - Else expected immediate reward:
            STEP_COST + p_cong_next * CONGESTION_PENALTY
          and expected next value:
            E[V(nx,ny,c')] where c' is random with p_cong_next
        """
        nx, ny = _move(x, y, a, grid_size)

        if (nx, ny) == goal:
            return STEP_COST + GOAL_REWARD

        r_exp = STEP_COST + (p_cong_next * CONGESTION_PENALTY)
        v_next = (1.0 - p_cong_next) * V[(nx, ny, 0)] + p_cong_next * V[(nx, ny, 1)]
        return r_exp + gamma * v_next

    def policy_evaluation() -> None:
        """Iteratively evaluate current policy pi until convergence."""
        for _ in range(eval_max_iters):
            delta = 0.0

            for x in range(grid_size):
                for y in range(grid_size):
                    for c in (0, 1):
                        if (x, y) == goal:
                            continue

                        old_v = V[(x, y, c)]
                        a = pi[(x, y, c)]
                        new_v = q_value_under_model(x, y, a)
                        V[(x, y, c)] = new_v
                        delta = max(delta, abs(old_v - new_v))

            if delta < eval_theta:
                break

    def policy_improvement() -> bool:
        """
        Improve policy greedily w.r.t. current V.
        Returns True if policy is stable (no changes), else False.
        """
        stable = True

        for x in range(grid_size):
            for y in range(grid_size):
                for c in (0, 1):
                    if (x, y) == goal:
                        pi[(x, y, c)] = STAY
                        continue

                    old_a = pi[(x, y, c)]

                    best_a = old_a
                    best_q = float("-inf")
                    for a in ACTIONS:
                        q = q_value_under_model(x, y, a)
                        if q > best_q:
                            best_q = q
                            best_a = a

                    pi[(x, y, c)] = best_a
                    if best_a != old_a:
                        stable = False

        return stable

    # Main loop: evaluate -> improve until stable
    for _ in range(improve_max_iters):
        policy_evaluation()
        if policy_improvement():
            break

    return V, pi


# Quick test
if __name__ == "__main__":
    V, pi = policy_iteration(grid_size=5, goal=(4, 4))
    print("V(start, no congestion) =", V[(0, 0, 0)])
    print("Best action at start =", pi[(0, 0, 0)])
