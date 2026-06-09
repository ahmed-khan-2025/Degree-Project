import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env.resource_env import ResourceEnv


# =========================================================
# CALLBACK: TRAINING LOG
# =========================================================

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self):
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_episode_reward += reward

        self.logs.append({
            "step": self.num_timesteps,
            "reward": reward,
            "done": done
        })

        if done:
            self.episode_rewards.append({
                "episode": len(self.episode_rewards),
                "episode_reward": self.current_episode_reward
            })
            self.current_episode_reward = 0

        return True

    def _on_training_end(self):
        os.makedirs("results", exist_ok=True)

        # step log
        pd.DataFrame(self.logs).to_csv(
            "results/training_log.csv",
            index=False
        )

        # episode log
        pd.DataFrame(self.episode_rewards).to_csv(
            "results/episode_log.csv",
            index=False
        )

        print("Saved logs (step + episode)")


# =========================================================
# ENVIRONMENT
# =========================================================

def make_env():
    return ResourceEnv("data/chambers.json")


env = DummyVecEnv([make_env])


# =========================================================
# PPO MODEL
# =========================================================

model = PPO(
    "MlpPolicy",
    env,

    learning_rate=3e-4,
    n_steps=8192,
    batch_size=256,

    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,

    ent_coef=0.03,

    verbose=1
)


# =========================================================
# TRAINING
# =========================================================

callback = RewardLoggerCallback()

model.learn(
    total_timesteps=300_000,
    callback=callback
)


# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs("models", exist_ok=True)
model.save("models/ppo_scheduler")

print("\nTraining completed successfully")