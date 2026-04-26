"""Deep Q-Network utilities for Gymnasium Taxi-v3 and scalable Taxi variants."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class DQNConfig:
    """Configuration for Taxi-v3 DQN training."""

    env_id: str = "Taxi-v3"
    seed: int = 42
    num_episodes: int = 3000
    max_steps: int = 200
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    buffer_size: int = 10000
    target_update_freq: int = 20
    hidden_dim: int = 128
    moving_average_window: int = 100
    device: str | None = None


def set_seed(seed: int) -> None:
    """Set random seeds used by NumPy, Python's random module, and PyTorch."""

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_device(device: str | None = None) -> torch.device:
    """Return a PyTorch device."""

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_state(state: int, n_states: int) -> np.ndarray:
    """One-hot encode a discrete state index."""

    x = np.zeros(n_states, dtype=np.float32)
    x[int(state)] = 1.0
    return x


class QNetwork(nn.Module):
    """Two-hidden-layer MLP used to approximate Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Simple replay buffer for off-policy DQN updates."""

    def __init__(self, capacity: int):
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(
    state_vec: np.ndarray,
    q_net: QNetwork,
    epsilon: float,
    n_actions: int,
    device: torch.device,
) -> int:
    """Select an action using epsilon-greedy exploration."""

    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))

    state_tensor = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def train_step(
    q_net: QNetwork,
    target_net: QNetwork,
    memory: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float | None:
    """Run one DQN gradient update."""

    if len(memory) < batch_size:
        return None

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    current_q = q_net(states_t).gather(1, actions_t)

    with torch.no_grad():
        next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
        target_q = rewards_t + gamma * next_q * (1 - dones_t)

    loss = F.mse_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train_dqn(config: DQNConfig | None = None, env: gym.Env | None = None, verbose: bool = True) -> dict[str, Any]:
    """Train a DQN agent.

    If `env` is omitted, Gymnasium Taxi-v3 is created from `config.env_id`.
    """

    config = config or DQNConfig()
    set_seed(config.seed)

    device = get_device(config.device)

    owns_env = env is None
    env = env or gym.make(config.env_id)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_net = QNetwork(n_states, n_actions, hidden_dim=config.hidden_dim).to(device)
    target_net = QNetwork(n_states, n_actions, hidden_dim=config.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=config.learning_rate)
    memory = ReplayBuffer(config.buffer_size)

    epsilon = config.epsilon_start
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    moving_avg_rewards: list[float] = []
    reward_window: deque[float] = deque(maxlen=config.moving_average_window)
    losses: list[float] = []

    if verbose:
        print("Number of states:", n_states)
        print("Number of actions:", n_actions)
        print("Device:", device)

    for episode in range(config.num_episodes):
        state, _ = env.reset(seed=config.seed + episode)
        state_vec = encode_state(state, n_states)

        total_reward = 0.0
        step_count = 0

        while step_count < config.max_steps:
            action = select_action(state_vec, q_net, epsilon, n_actions, device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_vec = encode_state(next_state, n_states)

            memory.push(state_vec, action, reward, next_state_vec, done)
            loss = train_step(
                q_net=q_net,
                target_net=target_net,
                memory=memory,
                optimizer=optimizer,
                batch_size=config.batch_size,
                gamma=config.gamma,
                device=device,
            )
            if loss is not None:
                losses.append(loss)

            state_vec = next_state_vec
            total_reward += reward
            step_count += 1

            if done:
                break

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        if (episode + 1) % config.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        reward_window.append(total_reward)
        moving_avg_rewards.append(float(np.mean(reward_window)))

        if verbose and (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {total_reward:7.1f} | "
                f"Avg({config.moving_average_window}): {moving_avg_rewards[-1]:7.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    if owns_env:
        env.close()

    return {
        "q_net": q_net,
        "episode_rewards": np.asarray(episode_rewards, dtype=np.float32),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int32),
        "moving_avg_rewards": np.asarray(moving_avg_rewards, dtype=np.float32),
        "losses": np.asarray(losses, dtype=np.float32),
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "network_params": int(sum(p.numel() for p in q_net.parameters())),
        "config": config,
        "device": device,
    }


def evaluate_dqn(
    q_net: QNetwork,
    env_id: str = "Taxi-v3",
    env: gym.Env | None = None,
    n_eval_episodes: int = 100,
    seed: int = 1000,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a DQN policy greedily."""

    device = get_device(str(device) if device is not None else None)
    q_net = q_net.to(device)
    q_net.eval()

    owns_env = env is None
    env = env or gym.make(env_id)

    n_states = env.observation_space.n
    rewards: list[float] = []
    lengths: list[int] = []

    for episode in range(n_eval_episodes):
        state, _ = env.reset(seed=seed + episode)

        total_reward = 0.0
        step_count = 0
        done = False

        while not done:
            state_vec = encode_state(state, n_states)
            state_tensor = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action = int(torch.argmax(q_net(state_tensor), dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        lengths.append(step_count)

    if owns_env:
        env.close()

    rewards_array = np.asarray(rewards, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.int32)

    metrics = {
        "avg_reward": float(rewards_array.mean()),
        "avg_steps": float(lengths_array.mean()),
        "rewards": rewards_array,
        "lengths": lengths_array,
    }

    if verbose:
        print(f"Average evaluation reward: {metrics['avg_reward']:.2f}")
        print(f"Average evaluation steps : {metrics['avg_steps']:.2f}")

    return metrics


def render_dqn_policy(
    q_net: QNetwork,
    env_id: str = "Taxi-v3",
    n_demo_episodes: int = 3,
    seed: int = 2000,
    device: torch.device | str | None = None,
) -> None:
    """Render a few greedy-policy DQN episodes in ANSI mode."""

    device = get_device(str(device) if device is not None else None)
    q_net = q_net.to(device)
    q_net.eval()

    env = gym.make(env_id, render_mode="ansi")
    n_states = env.observation_space.n

    for episode in range(n_demo_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        print(f"\n===== Demo Episode {episode + 1} =====")
        print(env.render())

        while not done:
            state_vec = encode_state(state, n_states)
            state_tensor = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action = int(torch.argmax(q_net(state_tensor), dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

            print(env.render())

        print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    result = train_dqn()
    evaluate_dqn(result["q_net"], device=result["device"])
