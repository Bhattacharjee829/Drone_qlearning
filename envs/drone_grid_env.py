# drone_grid_env.py


import random

class DroneGridEnv:
    def __init__(self, grid_size=5, max_steps=50):
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Fixed positions
        self.start = (0, 0)
        self.goal = (4, 4)

        self.reset()

    # -----------------------------
    # Required API
    # -----------------------------
    def reset(self):
        self.agent_pos = list(self.start)
        self.steps = 0

        # One congestion cell (moves randomly)
        self.congestion = self._random_cell()

        return self._get_state()

    def step(self, action):
        """
        action:
        0 = up
        1 = down
        2 = left
        3 = right
        """

        self.steps += 1

        # Move agent
        if action == 0:   # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1: # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2: # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)

        # Move congestion randomly
        self.congestion = self._random_cell()

        reward = -1  # step penalty
        done = False

        # Check congestion
        if tuple(self.agent_pos) == self.congestion:
            reward = -10

        # Check goal
        if tuple(self.agent_pos) == self.goal:
            reward = 10
            done = True

        # Max steps reached
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done


    def _get_state(self):
        """
        State = (x, y, congestion_flag)
        """
        x, y = self.agent_pos
        congestion_flag = int(tuple(self.agent_pos) == self.congestion)
        return (x, y, congestion_flag)

    def _random_cell(self):
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        return (x, y)



if __name__ == "__main__":
    env = DroneGridEnv()
    state = env.reset()
    done = False

    print("Initial state:", state)

    while not done:
        action = random.randint(0, 3)
        next_state, reward, done = env.step(action)
        print("Action:", action, "Next:", next_state, "Reward:", reward)
