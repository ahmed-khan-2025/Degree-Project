import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

import pandas as pd
from stable_baselines3 import PPO
from env.resource_env import ResourceEnv
from baseline.fcfs import run_fcfs

# =========================================================
# PPO
# =========================================================

env = ResourceEnv(
    "data/chambers.json"
)

model = PPO.load(
    "models/ppo_scheduler"
)

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

ppo_result = {

    "method": "PPO",
    "reward": env.get_total_reward(),
    "utilization": env.get_utilization(),
    "completion": env.get_completion_rate(),
    "reconfigs": env.get_reconfigs()
}

# =========================================================
# FCFS
# =========================================================

fcfs_result = run_fcfs()

# =========================================================
# PRINT
# =========================================================

print("\n===== PPO =====")
print(ppo_result)

print("\n===== FCFS =====")
print(fcfs_result)

# =========================================================
# SAVE CSV
# =========================================================

os.makedirs(
    "results",
    exist_ok=True
)

df = pd.DataFrame([
    ppo_result,
    fcfs_result
])

df.to_csv(
    "results/single_results.csv",
    index=False
)
print("\nSaved: results/single_results.csv")