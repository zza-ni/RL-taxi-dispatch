"""Tabular Q-learning utilities for Gymnasium Taxi-v3.

This module was separated from the original project notebook so that the
training and evaluation code can be reproduced from regular Python files.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import random


@dataclass
class QLearningConfig:
    """Configuration for Taxi-v3 tabular Q-learning."""

    env_id: str = "Taxi-v3"
    seed: int = 42
    num_episodes: int = 8000
    max_steps: int = 200
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9992
    moving_average_window: int = 100
    random_tie_breaking: bool = True
    save_best_q_table: bool = True


def set_seed(seed: int) -> None:
    """Set random seeds used by NumPy and Python's random module."""

    np.random.seed(seed)
    random.seed(seed)


def _argmax_action(q_values: np.ndarray, random_tie_breaking: bool = True) -> int:
    """Return a greedy action, optionally breaking argmax ties randomly."""

    if random_tie_breaking:
        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(np.random.choice(best_actions))
    return int(np.argmax(q_values))


def epsilon_greedy_action(
    state: int,
    q_table: np.ndarray,
    epsilon: float,
    n_actions: int,
    random_tie_breaking: bool = True,
) -> int:
    """Select an action using epsilon-greedy exploration."""

    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    return _argmax_action(q_table[state], random_tie_breaking=random_tie_breaking)


def train_q_learning(config: QLearningConfig | None = None, verbose: bool = True) -> dict[str, Any]:
    """Train a tabular Q-learning agent on Taxi-v3.

    Returns a dictionary containing the final Q-table, the best Q-table, and
    training history arrays.
    """

    config = config or QLearningConfig()
    set_seed(config.seed)

    env = gym.make(config.env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=np.float32)
    best_q_table: np.ndarray | None = None
    best_moving_avg = -np.inf

    epsilon = config.epsilon_start

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    moving_avg_rewards: list[float] = []
    reward_window: deque[float] = deque(maxlen=config.moving_average_window)

    if verbose:
        print("Number of states:", n_states)
        print("Number of actions:", n_actions)

    for episode in range(config.num_episodes):
        state, _ = env.reset(seed=config.seed + episode)
        total_reward = 0.0
        step_count = 0

        for _ in range(config.max_steps):
            action = epsilon_greedy_action(
                state=state,
                q_table=q_table,
                epsilon=epsilon,
                n_actions=n_actions,
                random_tie_breaking=config.random_tie_breaking,
            )

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next_q = np.max(q_table[next_state])
            td_target = reward + config.gamma * best_next_q * (1 - float(done))
            td_error = td_target - q_table[state, action]
            q_table[state, action] += config.alpha * td_error

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                break

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        reward_window.append(total_reward)

        current_mavg = float(np.mean(reward_window))
        moving_avg_rewards.append(current_mavg)

        if config.save_best_q_table and current_mavg > best_moving_avg:
            best_moving_avg = current_mavg
            best_q_table = q_table.copy()

        if verbose and (episode + 1) % 500 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {total_reward:7.1f} | "
                f"Avg({config.moving_average_window}): {current_mavg:7.2f} | "
                f"Epsilon: {epsilon:.4f}"
            )

    env.close()

    if best_q_table is None:
        best_q_table = q_table.copy()
        best_moving_avg = float(np.mean(reward_window)) if reward_window else -np.inf

    if verbose:
        print("Training finished.")
        print("Best moving average reward:", round(best_moving_avg, 2))

    return {
        "q_table": q_table,
        "best_q_table": best_q_table,
        "episode_rewards": np.asarray(episode_rewards, dtype=np.float32),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int32),
        "moving_avg_rewards": np.asarray(moving_avg_rewards, dtype=np.float32),
        "best_moving_avg": float(best_moving_avg),
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "config": config,
    }


def evaluate_q_learning(
    q_table: np.ndarray,
    env_id: str = "Taxi-v3",
    n_eval_episodes: int = 200,
    seed: int = 1000,
    random_tie_breaking: bool = True,
    high_quality_threshold: float = 8.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a learned Q-table with a greedy policy."""

    env = gym.make(env_id)
    rewards: list[float] = []
    lengths: list[int] = []

    for episode in range(n_eval_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action = _argmax_action(q_table[state], random_tie_breaking=random_tie_breaking)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        lengths.append(step_count)

    env.close()

    rewards_array = np.asarray(rewards, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.int32)

    metrics = {
        "avg_reward": float(rewards_array.mean()),
        "avg_steps": float(lengths_array.mean()),
        "success_ratio": float((rewards_array >= 0).mean()),
        "high_quality_ratio": float((rewards_array >= high_quality_threshold).mean()),
        "rewards": rewards_array,
        "lengths": lengths_array,
    }

    if verbose:
        print(f"Average evaluation reward: {metrics['avg_reward']:.2f}")
        print(f"Average evaluation steps : {metrics['avg_steps']:.2f}")
        print(f"Success ratio (reward >= 0): {metrics['success_ratio']:.3f}")
        print(
            f"High-quality episodes (reward >= {high_quality_threshold:g}): "
            f"{metrics['high_quality_ratio']:.3f}"
        )

    return metrics


def render_q_learning_policy(
    q_table: np.ndarray,
    env_id: str = "Taxi-v3",
    n_demo_episodes: int = 3,
    seed: int = 2000,
    random_tie_breaking: bool = True,
) -> None:
    """Render a few greedy-policy episodes in ANSI mode."""

    env = gym.make(env_id, render_mode="ansi")

    for episode in range(n_demo_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        print(f"\n===== Demo Episode {episode + 1} =====")
        print(env.render())

        while not done:
            action = _argmax_action(q_table[state], random_tie_breaking=random_tie_breaking)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

            print(env.render())

        print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    result = train_q_learning()
    evaluate_q_learning(result["best_q_table"])
