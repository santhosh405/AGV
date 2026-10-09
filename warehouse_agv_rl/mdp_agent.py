"""Value iteration planner for the warehouse MDP."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class ValueIterationAgent:
    """Bellman optimality solver for the warehouse environment."""

    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6, max_iterations: int = 1000) -> None:
        self.env = env
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.max_iterations = int(max_iterations)
        self.V = np.zeros(self.env.n_states, dtype=float)
        self.policy = np.zeros(self.env.n_states, dtype=int)
        self.delta_history: List[float] = []
        self.iteration_count = 0

    def solve(self) -> Tuple[np.ndarray, int]:
        transition_model = self.env.get_transition_model()
        for iteration in range(1, self.max_iterations + 1):
            new_values = self.V.copy()
            delta = 0.0
            for state in range(self.env.n_states):
                if self.env.is_goal_state(state):
                    new_values[state] = 0.0
                    continue
                best_value = -np.inf
                for action in range(self.env.n_actions):
                    action_value = 0.0
                    for prob, next_state, reward, done in transition_model[state][action]:
                        future_value = 0.0 if done else self.gamma * self.V[next_state]
                        action_value += prob * (reward + future_value)
                    if action_value > best_value:
                        best_value = action_value
                new_values[state] = best_value
                delta = max(delta, abs(new_values[state] - self.V[state]))
            self.V = new_values
            self.delta_history.append(delta)
            self.iteration_count = iteration
            if delta < self.theta:
                break
        self.policy = self.extract_policy()
        return self.V, self.iteration_count

    def extract_policy(self) -> np.ndarray:
        transition_model = self.env.get_transition_model()
        policy = np.zeros(self.env.n_states, dtype=int)
        for state in range(self.env.n_states):
            if self.env.is_goal_state(state):
                policy[state] = 0
                continue
            best_action = 0
            best_value = -np.inf
            for action in range(self.env.n_actions):
                action_value = 0.0
                for prob, next_state, reward, done in transition_model[state][action]:
                    future_value = 0.0 if done else self.gamma * self.V[next_state]
                    action_value += prob * (reward + future_value)
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            policy[state] = best_action
        return policy

    def get_action(self, state: int) -> int:
        if state < 0 or state >= self.env.n_states:
            raise ValueError("State out of range.")
        return int(self.policy[state])
