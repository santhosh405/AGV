# 🤖 Autonomous Warehouse AGV Navigation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pygame-Simulation-green?style=for-the-badge" alt="Pygame" />
  <img src="https://img.shields.io/badge/Algorithm-Q--Learning%20%7C%20MDP-blueviolet?style=for-the-badge" alt="Algorithms" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License" />
</p>

An intelligent path planning and collision avoidance system for **Automated Guided Vehicles (AGVs)** operating in structured fulfillment centers. This project benchmarks **Model-Based Planning (Value Iteration)** against **Model-Free Reinforcement Learning (Tabular Q-Learning)** across a custom discrete warehouse floor.

---

## 📌 Project Overview

In high-throughput e-commerce fulfillment hubs, autonomous rovers must navigate narrow aisles, circumvent storage racks, and reach target drop-off docks along the shortest possible route. 

This repository models warehouse floor dynamics as a discrete **Markov Decision Process (MDP)** and compares two core paradigms:

1. **Value Iteration (Model-Based / MDP):** Solves the environment with dynamic programming using the **Bellman Optimality Equation**, assuming known transition and reward structures.
2. **Tabular Q-Learning (Model-Free RL):** Learns an optimal action-value policy $\pi^*$ through trial-and-error interactions via **Temporal Difference (TD)** control and an $\epsilon$-greedy exploration schedule.

---

## ⚙️ Mathematical & MDP Formulation

The navigation grid is formalized as the tuple $\langle S, A, P, R, \gamma \rangle$:

| Parameter | Symbol | Specification |
| :--- | :---: | :--- |
| **State Space** | $S$ | Discrete 2D coordinates $(r, c)$ on an $N \times N$ matrix |
| **Action Space** | $A$ | 4 orthogonal movements: $\text{Up } (0), \text{Down } (1), \text{Left } (2), \text{Right } (3)$ |
| **Transitions** | $P$ | Deterministic movement bounded by grid boundaries |
| **Discount Factor** | $\gamma$ | $0.95$ (prioritizes sustained long-term returns) |

### Reward Dynamics ($R$)
$$\mathcal{R}(s) = \begin{cases} +100 & \text{if } s = \text{Goal Dock (terminal)} \\ -100 & \text{if } s \in \text{Storage Racks / Hazards (terminal)} \\ -1 & \text{for any valid step (living cost / shortest-path incentive)} \end{cases}$$

---

## 🔬 Core Algorithms

### 1. Value Iteration (Bellman Optimality)
Iterates over all grid states until the value changes converge below a threshold $\theta = 10^{-4}$:
$$V_{k+1}(s) = \max_{a \in A} \sum_{s'} P(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V_k(s') \right]$$

### 2. Q-Learning (Off-Policy TD Control)
Updates state-action values based on sampled reward feedback and expected future rewards:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$
* **Learning Rate ($\alpha$):** $0.1$
* **Exploration ($\epsilon$):** Starts at $1.0$, decaying multiplicatively by $0.995$ per episode down to $0.01$.

---

## 🛠️ Key Features

- **🎮 Real-Time 2D Simulation:** Interactive Pygame visualization showing the AGV traversing the shortest hazard-free route.
- **🗺️ Policy Vector Maps:** Seaborn/Matplotlib heatmaps displaying state-values overlaid with policy direction vectors ($\uparrow, \downarrow, \leftarrow, \rightarrow$).
- **📈 Convergence Tracking:** Episode-versus-reward plots tracking the learning trajectory from random collisions to steady convergence.

---

## 📂 Project Structure

```text
warehouse_agv_rl/
├── environment.py    # Custom Warehouse GridWorld environment logic
├── mdp_agent.py      # Value Iteration agent (Bellman solver)
├── q_agent.py        # Tabular Q-Learning agent (TD control)
├── visualizer.py     # Matplotlib policy heatmaps & reward curves
├── simulation.py     # Pygame real-time navigation interface
└── main.py           # Training pipeline and master execution script
