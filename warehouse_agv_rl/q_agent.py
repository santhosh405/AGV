"""Tabular Q-learning agent for the warehouse environment."""

from __future__ import annotations

from typing import List

import numpy as np


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning agent."""

    def __init__(
        self,
        env,
        alpha: float = 0.1,
        gamma: float = 0.99,
        episodes: int = 2000,
        max_steps_per_episode: int = 200,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.episodes = int(episodes)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.Q = np.zeros((self.env.n_states, self.env.n_actions), dtype=float)
        self.reward_history: List[float] = []
        self.steps_history: List[int] = []
        self.epsilon_history: List[float] = []
        self.episodes_trained = 0

    def select_action(self, state: int, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.env.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        target = r + (0.0 if done else self.gamma * np.max(self.Q[s2]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def train(self, env) -> List[float]:
        self.env = env
        self.Q = np.zeros((self.env.n_states, self.env.n_actions), dtype=float)
        self.reward_history = []
        self.steps_history = []
        self.epsilon_history = []
        self.episodes_trained = 0
        for episode in range(self.episodes):
            state = env.reset()
            episode_reward = 0.0
            steps = 0
            while steps < self.max_steps_per_episode:
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                steps += 1
                if done:
                    break
            self.reward_history.append(episode_reward)
            self.steps_history.append(steps)
            self.epsilon_history.append(self.epsilon)
            self.episodes_trained = episode + 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.reward_history

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1)
