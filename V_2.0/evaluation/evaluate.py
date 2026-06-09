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
# LOAD MODEL
# =========================================================
model = PPO.load("models/ppo_scheduler")

results = []

# =========================================================
# PPO
# =========================================================
for i in range(10):

    env = ResourceEnv("data/chambers.json")
    obs, _ = env.reset(seed=i)

    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    results.append({
        "run": i,
        "method": "PPO",
        "reward": float(env.get_total_reward()),
        "utilization": float(env.get_utilization()),
        "completion": float(env.get_completion_rate()),
        "reconfigs": int(env.get_reconfigs())
    })

# =========================================================
# FCFS
# =========================================================
for i in range(10):

    res = run_fcfs(seed=i)
    res["run"] = i
    res["method"] = "FCFS"

    results.append(res)

# =========================================================
# DATAFRAME
# =========================================================
df = pd.DataFrame(results)

# Save raw results
df.to_csv("results/results_raw.csv", index=False)

# =========================================================
# MEAN + STD SUMMARY
# =========================================================
summary = df.groupby("method").agg(
    reward_mean=("reward", "mean"),
    reward_std=("reward", "std"),

    utilization_mean=("utilization", "mean"),
    utilization_std=("utilization", "std"),

    completion_mean=("completion", "mean"),
    completion_std=("completion", "std"),

    reconfigs_mean=("reconfigs", "mean"),
    reconfigs_std=("reconfigs", "std"),
).reset_index()

# Save summary
summary.to_csv("results/results_summary.csv", index=False)

# =========================================================
# PRINT
# =========================================================
print("\n===== RAW RESULTS =====")
print(df)

print("\n===== SUMMARY (MEAN ± STD) =====")
print(summary)