# Comparative Analysis of Q-Learning and DQN for Intelligent Taxi Dispatching

**Team 4** | Dogyun Kwon, Iroo Jeon, Chanhee Jeong, Jiyeong Hwang  
Sungkyunkwan University

---

## Overview

This project compares two foundational Reinforcement Learning algorithms—**Tabular Q-Learning** and **Deep Q-Networks (DQN)**—applied to the taxi dispatching problem. The taxi dispatching task is formulated as a Markov Decision Process (MDP) within the [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment.

The key research question is: *how does each method scale as the state space grows?* We evaluate both algorithms on the standard 5×5 grid and extend experiments to a 15×15 grid to expose scalability trade-offs.

---

## Key Results

| Method | Avg. Eval Reward | Success Ratio | Environment |
|---|---|---|---|
| Q-Learning (basic) | 7.63 | — | Taxi-v3 (5×5) |
| Q-Learning (improved) | **7.96** | **1.000** | Taxi-v3 (5×5) |
| DQN | competitive | — | Taxi-v3 (5×5) |
| Q-Learning | −1,465.74 | — | Scaled (15×15) |
| DQN | **−900.00** | — | Scaled (15×15) |

**Takeaway:** Q-Learning is stronger in the small discrete setting. DQN degrades more gracefully as the environment scales up, thanks to neural function approximation.

---

## Environment

- Python 3.8+
- [Gymnasium](https://gymnasium.farama.org/) (Taxi-v3)
- PyTorch (for DQN)
- NumPy, Matplotlib

### Installation

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install gymnasium numpy matplotlib torch pandas
```

All experiments were run on **Google Colab**. No GPU is required for Q-Learning; a GPU runtime is recommended for DQN to speed up training.

---

## Repository Structure

```
experiment.ipynb                   # Main notebook containing all experiments
requirements.txt                   # Python dependencies
README.md                          # This file
src/
├── q_learning.py                  # Tabular Q-Learning implementation
├── dqn.py                         # DQN implementation
├── scalable_taxi_env.py           # Custom scaled Taxi environment (up to 15×15)
└── evaluate.py                    # Shared evaluation utilities
scripts/
├── run_q_learning.py              # Script to train & evaluate Q-Learning
├── run_dqn.py                     # Script to train & evaluate DQN
├── run_scalability.py             # Script for scalability experiments
└── run_result.txt                 # Sample run output
Final Project Presentation.pdf     # Slide deck
Final Project Tech Report.pdf      # Full technical report
```

The notebook is organized into the following sections:

1. **Q-Learning (basic)** — baseline tabular Q-learning on Taxi-v3
2. **Q-Learning (improved)** — adds random tie-breaking and best-model checkpointing
3. **DQN** — deep Q-network with experience replay and target network
4. **Scalability Experiment** — compares both methods on expanded grid sizes (up to 15×15)

---

## Running the Code

### Option 1: Google Colab (recommended)

Open the notebook directly in Colab:  
👉 [Open in Colab](https://colab.research.google.com/drive/1I8liIaPOVkActo9t3RhZv_6rTNfNAbJ_)

Run all cells in order from top to bottom. Each section is self-contained.

### Option 2: Local Jupyter

```bash
pip install jupyter
pip install -r requirements.txt
jupyter notebook experiment.ipynb
```

Then run cells in order.

### Option 3: Run individual scripts locally

```bash
# Q-Learning
python scripts/run_q_learning.py

# DQN
python scripts/run_dqn.py

# Scalability experiment
python scripts/run_scalability.py
```

---

## Reproducing Each Experiment

### Q-Learning (Basic)
- Section: **"Q Learning"** in the notebook
- Hyperparameters: `num_episodes=5000`, `alpha=0.1`, `gamma=0.99`, `epsilon_decay=0.999`
- Evaluation: greedy policy over 100 episodes

### Q-Learning (Improved)
- Section: **"Q Learning 개선"** in the notebook
- Hyperparameters: `num_episodes=8000`, `epsilon_decay=0.9992`
- Adds random tie-breaking when Q-values are equal
- Saves the best Q-table by 100-episode moving average reward
- Evaluation: greedy policy over 200 episodes

### DQN
- Section: **"DQN"** in the notebook
- Architecture: two fully connected hidden layers with ReLU activations
- Uses experience replay buffer and separate target network
- State encoded as one-hot vector
- Evaluation: greedy policy after training

### Scalability Experiment
- Section: **"Scalability"** in the notebook
- Expands the Taxi environment grid from 5×5 up to 15×15
- Runs both Q-Learning and DQN under a fixed training budget
- Compares final reward profiles across grid sizes

---

## Reproducibility

All random seeds are fixed for reproducibility:

```python
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```

Per-episode seeds are set as `env.reset(seed=SEED + episode)` during training and with deterministic offsets during evaluation.

---

## References

1. T. G. Dietterich, "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition," arXiv, 1999.
2. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.
3. X. Zhou et al., "A robust deep reinforcement learning approach to driverless taxi dispatching under uncertain demand," *Information Sciences*, vol. 646, 2023.
4. T. M. Rajeh et al., "A Clustering-Based Multi-Agent Reinforcement Learning Framework for Finer-Grained Taxi Dispatching," *IEEE Trans. Intelligent Transportation Systems*, vol. 25, no. 9, 2024.
