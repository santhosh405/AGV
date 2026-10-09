"""Warehouse grid world environment for AGV navigation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class WarehouseEnv:
    """Custom warehouse grid environment with stochastic motion."""

    EMPTY = 0
    SHELF = 1
    PICKUP = 2
    GOAL = 3
    HAZARD = 4

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTIONS = {
        ACTION_UP: (-1, 0),
        ACTION_DOWN: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_RIGHT: (0, 1),
    }

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        slip_prob: float = 0.1,
        grid: List[List[int]] | None = None,
        start: Tuple[int, int] | None = None,
        goal: Tuple[int, int] | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.slip_prob = float(slip_prob)
        self.grid = self._build_default_grid() if grid is None else [list(row) for row in grid]
        self.start_pos = start if start is not None else self._find_marker(self.PICKUP)
        self.goal_pos = goal if goal is not None else self._find_marker(self.GOAL)
        if self.start_pos is None:
            self.start_pos = (0, 0)
        if self.goal_pos is None:
            self.goal_pos = (rows - 1, cols - 1)
        self.current_state = self.pos_to_state(self.start_pos)
        self.goal_state = self.pos_to_state(self.goal_pos)

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def _build_default_grid(self) -> List[List[int]]:
        grid = [
            [self.PICKUP, 0, 0, 0, 0, 0, 0, 0],
            [0, self.SHELF, self.SHELF, 0, 0, self.SHELF, self.SHELF, 0],
            [0, 0, 0, 0, self.SHELF, 0, 0, 0],
            [0, self.SHELF, 0, 0, 0, 0, self.SHELF, 0],
            [0, self.SHELF, self.HAZARD, self.HAZARD, self.HAZARD, 0, 0, 0],
            [0, 0, 0, 0, self.SHELF, 0, 0, 0],
            [0, self.SHELF, 0, 0, 0, self.SHELF, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, self.GOAL],
        ]
        return grid

    def _find_marker(self, cell_type: int) -> Tuple[int, int] | None:
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == cell_type:
                    return row, col
        return None

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        row, col = pos
        return row * self.cols + col

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.cols)

    def reset(self) -> int:
        self.current_state = self.pos_to_state(self.start_pos)
        return self.current_state

    def is_goal_state(self, state: int) -> bool:
        return state == self.goal_state

    def _direction_from_action(self, action: int) -> Tuple[int, int]:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action {action}.")
        return self.ACTIONS[action]

    def _perpendicular_actions(self, action: int) -> List[int]:
        dirs = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]}
        return dirs[action]

    def _step_result(self, row: int, col: int, move: Tuple[int, int]) -> Tuple[int, int, float, bool]:
        dr, dc = move
        target_row = row + dr
        target_col = col + dc
        if not (0 <= target_row < self.rows and 0 <= target_col < self.cols):
            return row, col, -5.0, False
        if self.grid[target_row][target_col] == self.SHELF:
            return row, col, -5.0, False
        next_pos = (target_row, target_col)
        reward = -1.0
        if self.grid[target_row][target_col] == self.HAZARD:
            reward -= 10.0
        if next_pos == self.goal_pos:
            reward += 100.0
            return self.pos_to_state(next_pos), self.pos_to_state(next_pos), reward, True
        return self.pos_to_state(next_pos), self.pos_to_state(next_pos), reward, False

    def _transition_outcomes(self, state: int, action: int) -> List[Tuple[float, int, float, bool]]:
        row, col = self.state_to_pos(state)
        intended_move = self._direction_from_action(action)
        outcomes: List[Tuple[float, int, float, bool]] = []

        move_options = [(1.0 - self.slip_prob, intended_move)]
        for slip_action in self._perpendicular_actions(action):
            move_options.append((self.slip_prob / 2.0, self._direction_from_action(slip_action)))

        for prob, move in move_options:
            if prob <= 0.0:
                continue
            target_row = row + move[0]
            target_col = col + move[1]
            if not (0 <= target_row < self.rows and 0 <= target_col < self.cols):
                next_state = state
                reward = -5.0
                done = False
            elif self.grid[target_row][target_col] == self.SHELF:
                next_state = state
                reward = -5.0
                done = False
            else:
                next_state = self.pos_to_state((target_row, target_col))
                reward = -1.0
                if self.grid[target_row][target_col] == self.HAZARD:
                    reward -= 10.0
                if next_state == self.goal_state:
                    reward += 100.0
                    done = True
                else:
                    done = False
            outcomes.append((prob, next_state, reward, done))
        return outcomes

    def get_transition_model(self):
        model = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]
        for state in range(self.n_states):
            if self.is_goal_state(state):
                model[state] = [[(1.0, state, 0.0, True)] for _ in range(self.n_actions)]
                continue
            for action in range(self.n_actions):
                model[state][action] = self._transition_outcomes(state, action)
        return model

    def step(self, action: int):
        if self.current_state == self.goal_state:
            return self.current_state, 0.0, True, {"action": action, "terminal": True}
        if action not in range(self.n_actions):
            raise ValueError(f"Action must be in 0..{self.n_actions - 1}.")

        row, col = self.state_to_pos(self.current_state)
        move = self._direction_from_action(action)
        target_row = row + move[0]
        target_col = col + move[1]
        intended_choice = self._resolve_move(row, col, move)
        reward = -1.0
        done = False
        slip_roll = np.random.random()
        if slip_roll < self.slip_prob:
            slips = self._perpendicular_actions(action)
            slip_dir = slips[np.random.randint(0, len(slips))]
            next_row, next_col = self._resolve_move(row, col, self._direction_from_action(slip_dir))
        else:
            next_row, next_col = intended_choice

        # Apply penalties and terminal condition.
        if (next_row, next_col) == (row, col):
            reward += -5.0
        else:
            if self.grid[next_row][next_col] == self.HAZARD:
                reward += -10.0
        if (next_row, next_col) == self.goal_pos:
            reward += 100.0
            done = True
        next_state = self.pos_to_state((next_row, next_col))
        self.current_state = next_state
        info = {
            "action": action,
            "next_position": (next_row, next_col),
            "reward": reward,
            "done": done,
            "wall_hit": (next_row, next_col) == (row, col),
        }
        return next_state, reward, done, info

    def _resolve_move(self, row: int, col: int, move: Tuple[int, int]) -> Tuple[int, int]:
        dr, dc = move
        target_row = row + dr
        target_col = col + dc
        if not (0 <= target_row < self.rows and 0 <= target_col < self.cols):
            return row, col
        if self.grid[target_row][target_col] == self.SHELF:
            return row, col
        return target_row, target_col

    def render_text(self) -> str:
        lines: List[str] = []
        for row in range(self.rows):
            chars: List[str] = []
            for col in range(self.cols):
                cell = self.grid[row][col]
                if (row, col) == self.start_pos:
                    symbol = "S"
                elif (row, col) == self.goal_pos:
                    symbol = "G"
                elif cell == self.SHELF:
                    symbol = "#"
                elif cell == self.HAZARD:
                    symbol = "H"
                elif cell == self.PICKUP:
                    symbol = "P"
                else:
                    symbol = "."
                if self.pos_to_state((row, col)) == self.current_state:
                    if (row, col) != self.goal_pos:
                        symbol = "A"
                chars.append(symbol)
            lines.append(" ".join(chars))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"WarehouseEnv(rows={self.rows}, cols={self.cols}, slip_prob={self.slip_prob})"
