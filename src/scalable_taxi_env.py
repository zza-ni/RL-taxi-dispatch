"""Custom scalable Taxi environment for the final project.

The original Gymnasium Taxi-v3 has a fixed 5x5 map and 500 states. This
environment keeps the same core pickup/drop-off logic while allowing the grid
size to increase for scalability experiments.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces


class ScalableTaxiEnv(gym.Env):
    """Taxi-like environment with configurable N x N grid size."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size: int = 5, max_steps: int = 300, render_mode: str | None = None):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Passenger/destination locations: four corners.
        self.locs = [
            (0, 0),
            (0, grid_size - 1),
            (grid_size - 1, 0),
            (grid_size - 1, grid_size - 1),
        ]

        # state = taxi_row, taxi_col, passenger_loc(0~3, in_taxi=4), destination(0~3)
        self.n_passenger_states = len(self.locs) + 1
        self.n_destination_states = len(self.locs)

        self.observation_space = spaces.Discrete(
            grid_size * grid_size * self.n_passenger_states * self.n_destination_states
        )
        self.action_space = spaces.Discrete(6)  # south, north, east, west, pickup, dropoff

        self.state: int | None = None
        self.steps = 0

    def encode(self, taxi_row: int, taxi_col: int, passenger_loc: int, destination: int) -> int:
        """Encode state components into a single integer."""

        state = taxi_row
        state = state * self.grid_size + taxi_col
        state = state * self.n_passenger_states + passenger_loc
        state = state * self.n_destination_states + destination
        return int(state)

    def decode(self, state: int) -> tuple[int, int, int, int]:
        """Decode an integer state into state components."""

        destination = state % self.n_destination_states
        state //= self.n_destination_states

        passenger_loc = state % self.n_passenger_states
        state //= self.n_passenger_states

        taxi_col = state % self.grid_size
        state //= self.grid_size

        taxi_row = state
        return int(taxi_row), int(taxi_col), int(passenger_loc), int(destination)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment."""

        super().reset(seed=seed)
        self.steps = 0

        taxi_row = int(self.np_random.integers(0, self.grid_size))
        taxi_col = int(self.np_random.integers(0, self.grid_size))
        passenger_loc = int(self.np_random.integers(0, len(self.locs)))
        destination = int(self.np_random.integers(0, len(self.locs)))

        while destination == passenger_loc:
            destination = int(self.np_random.integers(0, len(self.locs)))

        self.state = self.encode(taxi_row, taxi_col, passenger_loc, destination)
        return int(self.state), {}

    def step(self, action: int):
        """Apply an action and return the Gymnasium step tuple."""

        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        taxi_row, taxi_col, passenger_loc, destination = self.decode(int(self.state))
        reward = -1
        terminated = False
        truncated = False

        if action == 0:  # south
            taxi_row = min(taxi_row + 1, self.grid_size - 1)
        elif action == 1:  # north
            taxi_row = max(taxi_row - 1, 0)
        elif action == 2:  # east
            taxi_col = min(taxi_col + 1, self.grid_size - 1)
        elif action == 3:  # west
            taxi_col = max(taxi_col - 1, 0)
        elif action == 4:  # pickup
            if passenger_loc < len(self.locs) and (taxi_row, taxi_col) == self.locs[passenger_loc]:
                passenger_loc = len(self.locs)  # in taxi
            else:
                reward = -10
        elif action == 5:  # dropoff
            if passenger_loc == len(self.locs) and (taxi_row, taxi_col) == self.locs[destination]:
                reward = 20
                terminated = True
            elif passenger_loc == len(self.locs) and (taxi_row, taxi_col) in self.locs:
                passenger_loc = self.locs.index((taxi_row, taxi_col))
                reward = -1
            else:
                reward = -10
        else:
            raise ValueError(f"Invalid action: {action}")

        self.state = self.encode(taxi_row, taxi_col, passenger_loc, destination)
        self.steps += 1

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        return int(self.state), reward, terminated, truncated, {}

    def render(self):
        """Render the current environment as a text grid."""

        if self.state is None:
            return ""

        taxi_row, taxi_col, passenger_loc, destination = self.decode(int(self.state))
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for idx, (row, col) in enumerate(self.locs):
            grid[row][col] = str(idx)

        if passenger_loc < len(self.locs):
            passenger_row, passenger_col = self.locs[passenger_loc]
            grid[passenger_row][passenger_col] = "P"

        destination_row, destination_col = self.locs[destination]
        grid[destination_row][destination_col] = "D"

        grid[taxi_row][taxi_col] = "T"

        return "\n".join(" ".join(row) for row in grid)
