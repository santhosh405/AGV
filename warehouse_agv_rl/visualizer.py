"""Plots for value functions, policies, and training curves."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _default_save_path(name: str) -> str:
    return os.path.join(_output_dir(), name)


def plot_value_heatmap(V, env, title: str, save_path: Optional[str] = None) -> str:
    """Create a heatmap of the value function with obstacles masked."""
    values = np.asarray(V).reshape(env.rows, env.cols)
    masked = np.where(np.asarray(env.grid) == env.SHELF, np.nan, values)
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(masked, cmap="viridis", origin="upper")
    for row in range(env.rows):
        for col in range(env.cols):
            if env.grid[row][col] == env.SHELF:
                ax.text(col, row, "#", ha="center", va="center", color="white", fontsize=10)
            elif not np.isnan(masked[row, col]):
                ax.text(col, row, f"{masked[row, col]:.1f}", ha="center", va="center", color="black", fontsize=9)
    ax.scatter(env.start_pos[1], env.start_pos[0], marker="s", s=120, color="blue", label="Start")
    ax.scatter(env.goal_pos[1], env.goal_pos[0], marker="*", s=180, color="green", label="Goal")
    for row in range(env.rows):
        for col in range(env.cols):
            if env.grid[row][col] == env.HAZARD:
                ax.text(col, row, "H", ha="center", va="center", color="orange", fontsize=10, weight="bold")
    ax.set_xticks(np.arange(env.cols))
    ax.set_yticks(np.arange(env.rows))
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.legend(loc="upper right")
    path = save_path or _default_save_path(f"{title.lower().replace(' ', '_')}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_policy_arrows(policy, env, title: str, save_path: Optional[str] = None) -> str:
    """Render a grid of action arrows for the policy."""
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    policy_grid = np.asarray(policy).reshape(env.rows, env.cols)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    for row in range(env.rows):
        for col in range(env.cols):
            cell = env.grid[row][col]
            if cell == env.SHELF:
                rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="black", edgecolor="black")
                ax.add_patch(rect)
            elif cell == env.HAZARD:
                rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="orange", edgecolor="black")
                ax.add_patch(rect)
            elif (row, col) == env.start_pos:
                rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="blue", edgecolor="black", alpha=0.5)
                ax.add_patch(rect)
            elif (row, col) == env.goal_pos:
                rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="green", edgecolor="black", alpha=0.7)
                ax.add_patch(rect)
            action = int(policy_grid[row, col])
            if cell != env.SHELF:
                ax.text(col, row, arrows.get(action, ""), ha="center", va="center", fontsize=16, weight="bold")
    for x in range(env.cols + 1):
        ax.axvline(x - 0.5, color="black", linewidth=0.8)
    for y in range(env.rows + 1):
        ax.axhline(y - 0.5, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(env.cols))
    ax.set_yticks(np.arange(env.rows))
    ax.set_title(title)
    path = save_path or _default_save_path(f"{title.lower().replace(' ', '_')}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_reward_curve(rewards, window: int = 50, title: str = "Reward curve", save_path: Optional[str] = None) -> str:
    """Plot raw and smoothed reward histories."""
    rewards = np.asarray(rewards, dtype=float)
    smooth = np.zeros_like(rewards)
    for idx in range(len(rewards)):
        start = max(0, idx - window + 1)
        smooth[idx] = rewards[start : idx + 1].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards, alpha=0.5, label="Raw reward")
    ax.plot(smooth, linewidth=2, label=f"{window}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    path = save_path or _default_save_path(f"{title.lower().replace(' ', '_')}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_convergence(delta_history, save_path: Optional[str] = None) -> str:
    """Plot the maximum Bellman residual per iteration."""
    deltas = np.asarray(delta_history, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(deltas) + 1), deltas, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max delta")
    ax.set_title("Value Iteration convergence")
    ax.grid(alpha=0.3)
    path = save_path or _default_save_path("value_iteration_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_comparison(mdp_policy, q_policy, V_mdp, V_q, env, save_path: Optional[str] = None) -> str:
    """Side-by-side comparison of the MDP and Q-learning policies."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    V_mdp_grid = np.asarray(V_mdp).reshape(env.rows, env.cols)
    V_q_grid = np.asarray(V_q).reshape(env.rows, env.cols)

    masked_mdp = np.where(np.asarray(env.grid) == env.SHELF, np.nan, V_mdp_grid)
    masked_q = np.where(np.asarray(env.grid) == env.SHELF, np.nan, V_q_grid)

    for index, values in enumerate([masked_mdp, masked_q]):
        ax = axes[index]
        image = ax.imshow(values, cmap="viridis", origin="upper")
        ax.set_title("Value Iteration" if index == 0 else "Q-Learning")
        for row in range(env.rows):
            for col in range(env.cols):
                if env.grid[row][col] == env.SHELF:
                    ax.text(col, row, "#", ha="center", va="center", color="white", fontsize=9)
        ax.scatter(env.start_pos[1], env.start_pos[0], marker="s", s=120, color="blue")
        ax.scatter(env.goal_pos[1], env.goal_pos[0], marker="*", s=180, color="green")
        ax.set_xticks(np.arange(env.cols))
        ax.set_yticks(np.arange(env.rows))
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    path = save_path or _default_save_path("policy_comparison.png")
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
