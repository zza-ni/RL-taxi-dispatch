"""Evaluation, plotting, and scalability-comparison utilities."""

from __future__ import annotations

from collections import deque
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from .dqn import QNetwork, ReplayBuffer, encode_state, get_device
    from .scalable_taxi_env import ScalableTaxiEnv
except ImportError:  # Allows running this file directly from inside src/
    from dqn import QNetwork, ReplayBuffer, encode_state, get_device
    from scalable_taxi_env import ScalableTaxiEnv


def plot_training_rewards(
    episode_rewards: np.ndarray | list[float],
    moving_avg_rewards: np.ndarray | list[float],
    title: str = "Training Reward",
) -> None:
    """Plot episode rewards and moving average rewards."""

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.4, label="Episode Reward")
    plt.plot(moving_avg_rewards, linewidth=2, label="Moving Avg Reward (100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_episode_lengths(
    episode_lengths: np.ndarray | list[int],
    title: str = "Episode Length",
) -> None:
    """Plot episode lengths."""

    plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths, alpha=0.6, label="Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_eval_distribution(
    values: np.ndarray | list[float],
    xlabel: str,
    title: str,
    bins: int = 20,
) -> None:
    """Plot a histogram for evaluation rewards or steps."""

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def train_q_learning_scalable(
    grid_size: int = 5,
    num_episodes: int = 2000,
    max_steps: int | None = None,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
) -> dict[str, Any]:
    """Train tabular Q-learning on ScalableTaxiEnv."""

    env = ScalableTaxiEnv(grid_size=grid_size, max_steps=max_steps or grid_size * grid_size * 4)

    np.random.seed(seed)
    random.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=np.float32)
    reward_window: deque[float] = deque(maxlen=100)
    moving_avg: list[float] = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0

        for _ in range(env.max_steps):
            if np.random.rand() < epsilon:
                action = int(np.random.randint(n_actions))
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next_q = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_q * (1 - float(done))
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_window.append(total_reward)
        moving_avg.append(float(np.mean(reward_window)))

    return {
        "env": env,
        "q_table": q_table,
        "moving_avg_rewards": np.asarray(moving_avg, dtype=np.float32),
        "final_avg_reward": float(np.mean(reward_window)),
        "qtable_entries": int(q_table.size),
    }


def evaluate_q_learning_scalable(
    env: ScalableTaxiEnv,
    q_table: np.ndarray,
    n_eval_episodes: int = 100,
    seed: int = 1234,
) -> tuple[float, float]:
    """Evaluate a Q-table on ScalableTaxiEnv."""

    rewards: list[float] = []
    lengths: list[int] = []

    for episode in range(n_eval_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        step_count = 0

        for _ in range(env.max_steps):
            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(step_count)

    return float(np.mean(rewards)), float(np.mean(lengths))


def train_dqn_scalable(
    grid_size: int = 5,
    num_episodes: int = 2000,
    max_steps: int | None = None,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    buffer_size: int = 10000,
    target_update_freq: int = 20,
    hidden_dim: int = 128,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train DQN on ScalableTaxiEnv."""

    env = ScalableTaxiEnv(grid_size=grid_size, max_steps=max_steps or grid_size * grid_size * 4)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    torch_device = get_device(device)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_net = QNetwork(n_states, n_actions, hidden_dim=hidden_dim).to(torch_device)
    target_net = QNetwork(n_states, n_actions, hidden_dim=hidden_dim).to(torch_device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(buffer_size)

    reward_window: deque[float] = deque(maxlen=100)
    moving_avg: list[float] = []

    def select_action_local(state_vec: np.ndarray, epsilon_now: float) -> int:
        if np.random.rand() < epsilon_now:
            return int(np.random.randint(n_actions))

        state_tensor = torch.as_tensor(state_vec, dtype=torch.float32, device=torch_device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step_local() -> None:
        if len(memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=torch_device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=torch_device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=torch_device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=torch_device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=torch_device).unsqueeze(1)

        current_q = q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + gamma * next_q * (1 - dones_t)

        loss = F.mse_loss(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        state_vec = encode_state(state, n_states)

        total_reward = 0.0

        for _ in range(env.max_steps):
            action = select_action_local(state_vec, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_vec = encode_state(next_state, n_states)
            memory.push(state_vec, action, reward, next_state_vec, done)
            train_step_local()

            state_vec = next_state_vec
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        reward_window.append(total_reward)
        moving_avg.append(float(np.mean(reward_window)))

    return {
        "env": env,
        "q_net": q_net,
        "moving_avg_rewards": np.asarray(moving_avg, dtype=np.float32),
        "final_avg_reward": float(np.mean(reward_window)),
        "network_params": int(sum(p.numel() for p in q_net.parameters())),
        "state_dim": int(n_states),
        "device": torch_device,
    }


def evaluate_dqn_scalable(
    env: ScalableTaxiEnv,
    q_net: QNetwork,
    n_eval_episodes: int = 100,
    seed: int = 1234,
    device: str | None = None,
) -> tuple[float, float]:
    """Evaluate a DQN model on ScalableTaxiEnv."""

    torch_device = get_device(device)
    q_net = q_net.to(torch_device)
    q_net.eval()

    rewards: list[float] = []
    lengths: list[int] = []

    n_states = env.observation_space.n

    for episode in range(n_eval_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        step_count = 0

        for _ in range(env.max_steps):
            state_vec = torch.as_tensor(
                encode_state(state, n_states),
                dtype=torch.float32,
                device=torch_device,
            ).unsqueeze(0)

            with torch.no_grad():
                action = int(torch.argmax(q_net(state_vec), dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(step_count)

    return float(np.mean(rewards)), float(np.mean(lengths))


def run_scalability_comparison(
    grid_sizes: list[int] | tuple[int, ...] = (5, 10, 15),
    num_episodes: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Q-learning vs. DQN scalability comparison and return a DataFrame."""

    comparison_results: list[dict[str, Any]] = []

    for grid_size in grid_sizes:
        print(f"\n===== Grid Size: {grid_size}x{grid_size} =====")

        q_result = train_q_learning_scalable(
            grid_size=grid_size,
            num_episodes=num_episodes,
            seed=seed,
        )
        q_eval_reward, q_eval_steps = evaluate_q_learning_scalable(
            q_result["env"],
            q_result["q_table"],
        )

        dqn_result = train_dqn_scalable(
            grid_size=grid_size,
            num_episodes=num_episodes,
            seed=seed,
        )
        dqn_eval_reward, dqn_eval_steps = evaluate_dqn_scalable(
            dqn_result["env"],
            dqn_result["q_net"],
        )

        comparison_results.append(
            {
                "grid_size": f"{grid_size}x{grid_size}",
                "num_states": q_result["env"].observation_space.n,
                "Q-learning avg reward": round(q_eval_reward, 2),
                "Q-learning avg steps": round(q_eval_steps, 2),
                "Q-table entries": q_result["qtable_entries"],
                "DQN avg reward": round(dqn_eval_reward, 2),
                "DQN avg steps": round(dqn_eval_steps, 2),
                "DQN state dim": dqn_result["state_dim"],
                "DQN params": dqn_result["network_params"],
            }
        )

    return pd.DataFrame(comparison_results)


def plot_q_table_growth(comparison_df: pd.DataFrame) -> None:
    """Plot Q-table entries as map size increases."""

    plt.figure(figsize=(8, 5))
    plt.plot(comparison_df["grid_size"], comparison_df["Q-table entries"], marker="o")
    plt.xlabel("Map Size")
    plt.ylabel("Number of Q-table Entries")
    plt.title("Scalability of Q-table as Map Size Increases")
    plt.grid(True)
    plt.show()


def plot_reward_comparison(comparison_df: pd.DataFrame) -> None:
    """Plot Q-learning and DQN average rewards by map size."""

    plt.figure(figsize=(8, 5))
    plt.plot(
        comparison_df["grid_size"],
        comparison_df["Q-learning avg reward"],
        marker="o",
        label="Q-learning",
    )
    plt.plot(
        comparison_df["grid_size"],
        comparison_df["DQN avg reward"],
        marker="o",
        label="DQN",
    )
    plt.xlabel("Map Size")
    plt.ylabel("Average Evaluation Reward")
    plt.title("Reward Comparison by Map Size")
    plt.legend()
    plt.grid(True)
    plt.show()
