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

## 📦 Installation , Train and Evaluate

Install dependencies:

```bash
pip install -r requirements.txt

🏋️ Training

Train the PPO model:

python train/train.py

The trained model will be saved in:

models/
📊 Evaluation

Run PPO evaluation:

python evaluation/run_ppo.py

Run baseline comparison:

python evaluation/fcfs.py
📁 Outputs

After running the project, you will get:

Trained model → models/
Training logs → results/
Evaluation results → printed in terminal or saved CSV (if enabled)
📌 Notes
Ensure data/chambers.json exists before training.
Train the model before running evaluation.
Results may vary slightly due to randomness in environment and training.