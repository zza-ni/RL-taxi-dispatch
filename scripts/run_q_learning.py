"""Run Taxi-v3 tabular Q-learning training and evaluation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q_learning import QLearningConfig, evaluate_q_learning, train_q_learning


def main() -> None:
    config = QLearningConfig()
    result = train_q_learning(config, verbose=True)
    evaluate_q_learning(result["best_q_table"], n_eval_episodes=200, verbose=True)


if __name__ == "__main__":
    main()
