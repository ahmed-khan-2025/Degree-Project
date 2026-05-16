# Degree-Project
# PPO-Based Resource Scheduling System

This project implements a reinforcement learning-based scheduling system for optimizing resource allocation in a simulated battery testing laboratory environment.

The system uses Proximal Policy Optimization (PPO) to improve scheduling decisions compared to baseline heuristics.

---

## 🚀 Features

- Custom Gymnasium environment for resource scheduling
- PPO agent using Stable-Baselines3
- Baseline comparison (Random / FCFS-style policy)
- Metrics:
  - Total reward
  - Utilization
  - Completion rate
  - Reconfiguration count
- Training and evaluation pipeline

---

## 🧠 Method

The environment simulates:

- Multiple testing chambers
- Job queue with random tasks
- Constraints such as current requirements and job duration

The agent selects scheduling strategies to:
- Maximize utilization
- Minimize reconfigurations
- Improve throughput

---

## 📦 Technologies

- Python
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- Pandas

---

## 📦 Installation

Install dependencies:

```bash
pip install -r requirements.txt

---
## 🏋️ Training

Run PPO training:

python train/train.py

The trained model will be saved automatically in the models/ folder.

📊 Evaluation

Run PPO evaluation:

python evaluation/run_ppo.py

Run baseline comparison:

python evaluation/fcfs.py
📁 Output
Trained model → models/
Logs/results → results/

🚀 Notes
Make sure data/chambers.json exists before running.
Train first before evaluation.
