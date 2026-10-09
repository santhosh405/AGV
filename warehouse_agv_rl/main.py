# Run with: cd warehouse_agv_rl && python main.py

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

import numpy as np

from environment import WarehouseEnv
from mdp_agent import ValueIterationAgent
from q_agent import QLearningAgent
from simulation import run_simulation
from visualizer import (
    plot_comparison,
    plot_convergence,
    plot_policy_arrows,
    plot_reward_curve,
    plot_value_heatmap,
)


def evaluate_policy(env: WarehouseEnv, policy, max_steps: int = 500):
    """Follow a policy deterministically from the warehouse start state."""
    eval_env = WarehouseEnv(
        rows=env.rows,
        cols=env.cols,
        slip_prob=0.0,
        grid=[row[:] for row in env.grid],
        start=env.start_pos,
        goal=env.goal_pos,
    )
    state = eval_env.reset()
    total_reward = 0.0
    path = [state]
    for _ in range(max_steps):
        action = int(policy[state])
        next_state, reward, done, _ = eval_env.step(action)
        total_reward += reward
        state = next_state
        path.append(state)
        if done:
            break
    return len(path) - 1, total_reward, path


def print_policy_agreement(mdp_policy, q_policy) -> float:
    """Return and print the percentage agreement between policies."""
    agreement = float(np.mean(mdp_policy == q_policy) * 100.0)
    print(f"{'Policy agreement':<28}{agreement:>8.2f}%")
    return agreement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and compare AGV RL agents.")
    parser.add_argument("--episodes", type=int, default=2000, help="Episodes for Q-learning.")
    parser.add_argument("--slip", type=float, default=0.1, help="Slip probability.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--simulate", choices=["mdp", "q", "none"], default="none", help="Agent to simulate after training.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    env = WarehouseEnv(slip_prob=args.slip)

    mdp_start = time.perf_counter()
    mdp_agent = ValueIterationAgent(env, gamma=args.gamma, max_iterations=2000)
    mdp_values, mdp_iterations = mdp_agent.solve()
    mdp_time = time.perf_counter() - mdp_start
    mdp_policy = mdp_agent.extract_policy()

    q_start = time.perf_counter()
    q_agent = QLearningAgent(env, alpha=args.alpha, gamma=args.gamma, episodes=args.episodes)
    q_agent.train(env)
    q_time = time.perf_counter() - q_start
    q_policy = q_agent.get_policy()

    mdp_path_len, mdp_reward, _ = evaluate_policy(env, mdp_policy)
    q_path_len, q_reward, _ = evaluate_policy(env, q_policy)
    agreement = print_policy_agreement(mdp_policy, q_policy)

    print("\nComparison Summary")
    print("-" * 120)
    print(f"{'Metric':<35}{'Value Iteration':>18}{'Q-Learning':>18}{'Agreement':>12}")
    print("-" * 120)
    print(f"{'Iterations / Episodes':<35}{mdp_iterations:>18}{q_agent.episodes_trained:>18}{'':>12}")
    print(f"{'Training time (s)':<35}{mdp_time:>18.4f}{q_time:>18.4f}{'':>12}")
    print(f"{'Path length to goal':<35}{mdp_path_len:>18}{q_path_len:>18}{'':>12}")
    print(f"{'Final greedy reward':<35}{mdp_reward:>18.2f}{q_reward:>18.2f}{'':>12}")
    print(f"{'Policy agreement %':<35}{'':>18}{'':>18}{agreement:>12.2f}")
    print("-" * 120)

    if not args.no_plots:
        plot_value_heatmap(mdp_values, env, "Value Iteration Value Map")
        plot_policy_arrows(mdp_policy, env, "Value Iteration Policy")
        plot_policy_arrows(q_policy, env, "Q-Learning Policy")
        plot_reward_curve(q_agent.reward_history, title="Q-Learning Reward Curve")
        plot_convergence(mdp_agent.delta_history)
        plot_comparison(mdp_policy, q_policy, mdp_values, q_agent.get_value_function(), env)
        print(f"Plots saved to: {outputs_dir}")

    if args.simulate == "mdp":
        run_simulation(env, mdp_policy, fps=5, title="Warehouse AGV - Value Iteration")
    elif args.simulate == "q":
        run_simulation(env, q_policy, fps=5, title="Warehouse AGV - Q-Learning")


if __name__ == "__main__":
    main()
