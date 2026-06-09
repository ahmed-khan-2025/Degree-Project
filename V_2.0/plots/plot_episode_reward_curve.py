import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots/output", exist_ok=True)

# Load episode data
log = pd.read_csv("results/episode_log.csv")

# smoothing
log["moving_avg"] = log["episode_reward"].rolling(window=20, min_periods=1).mean()

plt.figure(figsize=(8, 5))

plt.plot(
    log["episode"],
    log["episode_reward"],
    alpha=0.3,
    label="Episode Reward"
)

plt.plot(
    log["episode"],
    log["moving_avg"],
    linewidth=2,
    label="Smoothed Reward"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward per Episode")
plt.title("PPO Episode Reward Curve")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("plots/output/episode_reward_curve.png", dpi=300)
plt.close()

print("Episode reward curve saved")