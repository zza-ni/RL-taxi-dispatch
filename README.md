# 🚕 RL Taxi Dispatch

강화학습(Q-Learning / DQN)을 활용한 택시 배차 최적화 프로젝트입니다.  
기존 노트북 코드를 모듈화하여 Q-learning, DQN, ScalableTaxiEnv, 평가·시각화·확장성 실험 코드를 분리했습니다.  
Q-learning vs DQN vs Scalability 비교 구조는 기존 발표 내용에 맞춰 구성되었습니다.

---

## 📁 프로젝트 구조

```
RL-taxi-dispatch/
├── README.md
├── requirements.txt
├── experiment.ipynb                 # 기존 원본 노트북 보존
├── experiment_modularized.ipynb     # 모듈화된 .py 파일을 import해서 실행하는 노트북
├── src/
│   ├── q_learning.py
│   ├── dqn.py
│   ├── scalable_taxi_env.py
│   └── evaluate.py
└── scripts/
    ├── run_q_learning.py
    ├── run_dqn.py
    └── run_scalability.py
```

---

## 🗂️ 핵심 파일 역할

| 파일 | 역할 |
|------|------|
| `src/q_learning.py` | Taxi-v3 Q-learning 학습, 평가, 렌더링 |
| `src/dqn.py` | DQN network, replay buffer, 학습, 평가, 렌더링 |
| `src/scalable_taxi_env.py` | grid size 확장 가능한 custom Taxi 환경 |
| `src/evaluate.py` | 시각화, scalability comparison, scalable env용 Q-learning/DQN |
| `scripts/run_q_learning.py` | Q-learning 단독 실행 |
| `scripts/run_dqn.py` | DQN 단독 실행 |
| `scripts/run_scalability.py` | 5×5, 10×10, 15×15 비교 실험 실행 |
| `experiment_modularized.ipynb` | 모듈화된 .py 파일을 import해서 실행하는 얇은 노트북 |
| `experiment.ipynb` | 기존 원본 노트북 보존 |

---

## ⚙️ 환경 설정

```bash
pip install -r requirements.txt
```

**주요 의존성**

| 패키지 | 버전 |
|--------|------|
| `numpy` | >= 1.24 |
| `gymnasium` | >= 0.29 |
| `matplotlib` | >= 3.7 |
| `torch` | >= 2.0 |
| `pandas` | >= 2.0 |

---

## 🚀 실행 방법

### Q-Learning

```bash
python scripts/run_q_learning.py
```

### DQN

```bash
python scripts/run_dqn.py
```

### 확장성 비교 실험 (5×5 / 10×10 / 15×15)

```bash
python scripts/run_scalability.py
```

### 노트북 실행

```bash
jupyter notebook experiment_modularized.ipynb
```

---

## 🧠 알고리즘 개요

### Q-Learning
테이블 기반 강화학습. Taxi-v3 환경에서 상태-행동 가치 함수를 직접 업데이트합니다.

### DQN (Deep Q-Network)
신경망 기반 강화학습. Experience Replay와 Target Network를 활용하여 학습 안정성을 높입니다.

### Scalable Taxi Env
grid 크기를 동적으로 조절할 수 있는 custom Gymnasium 환경으로, 알고리즘 확장성 비교에 활용됩니다.
