import sys
import os
import pandas as pd
from stable_baselines3 import PPO

# =========================================================
# PATH SETUP
# =========================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from env.resource_env import ResourceEnv
from baseline.fcfs import run_fcfs

# =========================================================
# CONFIG
# =========================================================

RUNS = 10         # set to 1 for single-run behavior
SEED = 42
MODEL_PATH = "models/ppo_scheduler"

model = PPO.load(MODEL_PATH)

all_results = []

# =========================================================
# PPO EVALUATION
# =========================================================

for i in range(RUNS):

    env = ResourceEnv("data/chambers.json")
    obs, _ = env.reset(seed=SEED + i)

    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    all_results.append({
        "run": i,
        "method": "PPO",
        "reward": env.get_total_reward(),
        "utilization": env.get_utilization(),
        "completion": env.get_completion_rate(),
        "reconfigs": env.get_reconfigs()
    })

# =========================================================
# FCFS EVALUATION
# =========================================================

for i in range(RUNS):

    result = run_fcfs(seed=SEED + i)
    result["run"] = i
    result["method"] = "FCFS"

    all_results.append(result)

# =========================================================
# DATAFRAME + SAVE RAW RESULTS
# =========================================================

os.makedirs("results", exist_ok=True)

all_runs_df = pd.DataFrame(all_results)
all_runs_df.to_csv("results/all_runs.csv", index=False)

# =========================================================
# SINGLE-RUN PRINT (optional compatibility)
# =========================================================

if RUNS == 1:
    ppo_result = all_runs_df[all_runs_df["method"] == "PPO"].iloc[0].to_dict()
    fcfs_result = all_runs_df[all_runs_df["method"] == "FCFS"].iloc[0].to_dict()

    print("\n===== PPO =====")
    print(ppo_result)

    print("\n===== FCFS =====")
    print(fcfs_result)

    print("\nSaved: results/all_runs.csv")

# =========================================================
# SUMMARY (only meaningful if RUNS > 1)
# =========================================================

summary_rows = []

for method in ["PPO", "FCFS"]:
    df = all_runs_df[all_runs_df["method"] == method]

    summary_rows.append({
        "method": method,
        "reward_mean": df["reward"].mean(),
        "reward_std": df["reward"].std(),
        "utilization_mean": df["utilization"].mean(),
        "utilization_std": df["utilization"].std(),
        "completion_mean": df["completion"].mean(),
        "completion_std": df["completion"].std(),
        "reconfigs_mean": df["reconfigs"].mean(),
        "reconfigs_std": df["reconfigs"].std()
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results/summary_results.csv", index=False)

print("\n===== SUMMARY =====")
print(summary_df)

print("\nSaved:")
print("results/all_runs.csv")
print("results/summary_results.csv")