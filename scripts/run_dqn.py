"""Run Taxi-v3 DQN training and evaluation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqn import DQNConfig, evaluate_dqn, train_dqn


def main() -> None:
    config = DQNConfig()
    result = train_dqn(config, verbose=True)
    evaluate_dqn(result["q_net"], n_eval_episodes=100, device=result["device"], verbose=True)


if __name__ == "__main__":
    main()
