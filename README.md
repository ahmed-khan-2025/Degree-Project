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

---

## 🧠 Method

The environment simulates:

- Multiple testing chambers
- Job queue with randomly generated tasks
- Constraints such as required current, priority, and duration

The agent selects scheduling strategies to:
- Maximize utilization
- Minimize reconfigurations
- Improve overall efficiency

---

## 📦 Technologies

- Python
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- Pandas

---

## ⚙️ Installation

pip install -r requirements.txt

---

## 🏋️ Training

python train/train.py

models/

---

## 📊 Evaluation

python evaluation/run_ppo.py

python evaluation/fcfs.py

---

## 📁 Outputs

- models/ → trained model
- results/ → logs
- terminal → evaluation output

---

## 📌 Notes

Ensure data/chambers.json exists before running.  
Train before evaluation.  
Results may vary due to randomness.