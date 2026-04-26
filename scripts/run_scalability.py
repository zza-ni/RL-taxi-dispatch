"""Run scalability comparison between Q-learning and DQN."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import run_scalability_comparison


def main() -> None:
    comparison_df = run_scalability_comparison(grid_sizes=(5, 10, 15), num_episodes=2000, seed=42)
    print(comparison_df)


if __name__ == "__main__":
    main()
