# mdp/states.py
# Definition of state for the drone grid environment (Version 1)

from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    """
    State definition:
    x : drone x-position on the grid
    y : drone y-position on the grid
    c : congestion flag (1 if drone is in congestion, else 0)
    """
    x: int
    y: int
    c: int  # congestion flag (0 or 1)

    def as_tuple(self):
        return (self.x, self.y, self.c)

    def to_index(self, grid_size: int):
        """
        Convert state to a unique integer index
        Useful for tabular Q-learning and DP
        """
        return (self.x * grid_size + self.y) * 2 + self.c
