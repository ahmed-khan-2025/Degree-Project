import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================
# OUTPUT FOLDER
# =============================

os.makedirs("plots/output", exist_ok=True)

# =============================
# LOAD SUMMARY DATA
# =============================

results = pd.read_csv("results/results_summary.csv")

# =============================
# METRICS CONFIG
# =============================

metrics = [
    "reward",
    "utilization",
    "completion",
    "reconfigs"
]

titles = [
    "Total Reward",
    "Resource Utilization",
    "Task Completion Rate",
    "Reconfiguration Count"
]

mean_cols = [
    "reward_mean",
    "utilization_mean",
    "completion_mean",
    "reconfigs_mean"
]

std_cols = [
    "reward_std",
    "utilization_std",
    "completion_std",
    "reconfigs_std"
]

# =============================
# IEEE STYLE SETTINGS
# =============================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

# =============================
# CREATE SUBPLOTS
# =============================

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

methods = results["method"]
colors = ["#4D4D4D", "#A0A0A0"]

# =============================
# PLOTTING (MEAN ± STD)
# =============================

for i in range(len(metrics)):

    axes[i].bar(
        methods,
        results[mean_cols[i]],
        yerr=results[std_cols[i]],
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=1.0
    )

    axes[i].set_title(titles[i], pad=10)
    axes[i].set_xlabel("Scheduling Method")
    axes[i].set_ylabel(titles[i])

    axes[i].grid(axis="y", linestyle="--", alpha=0.5)

# =============================
# GLOBAL TITLE
# =============================

fig.suptitle(
    "Performance Comparison Between PPO and FCFS Scheduling",
    #fontweight="bold"
)

# =============================
# LAYOUT FIX
# =============================

plt.tight_layout(rect=[0, 0.03, 1, 1])

# =============================
# SAVE FIGURE
# =============================

plt.savefig(
    "plots/output/ppo_vs_fcfs_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: PPO vs FCFS summary figure")