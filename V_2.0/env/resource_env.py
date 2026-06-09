import gymnasium as gym
import numpy as np
import json
import random


class ResourceEnv(gym.Env):

    def __init__(self, data_path, max_steps=2000):

        super().__init__()

        with open(data_path, "r") as f:
            self.chambers = json.load(f)

        self.n_ch = len(self.chambers)

        self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )

        self.max_steps = max_steps
        self.reset()

    # -----------------------------
    # JOB GENERATION
    # -----------------------------
    def _new_job(self):
        return {
            "type": random.choice(["Cycling", "Performance"]),
            "duration": random.randint(20, 120),
            "required_current": random.choice([150, 200, 250, 300, 400]),
            "priority": random.randint(1, 5),
            "wait": 0
        }

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.t = 0
        self.completed = 0
        self.reconfigs = 0
        self.dropped = 0
        self.total_jobs = 0
        self.total_reward = 0.0

        for c in self.chambers:
            c["busy_until"] = 0

        self.queue = [self._new_job() for _ in range(30)]

        return self._obs(), {}

    # -----------------------------
    # OBSERVATION
    # -----------------------------
    def _obs(self):

        busy = sum(c["busy_until"] > self.t for c in self.chambers)
        job = self.queue[0]

        return np.array([
            busy / self.n_ch,
            len(self.queue) / 30,
            job["duration"] / 120,
            job["required_current"] / 400,
            job["priority"] / 5,
            self.completed / max(1, self.total_jobs),
            self.dropped / max(1, self.total_jobs),
            self.reconfigs / max(1, self.total_jobs),
            np.mean([max(0, c["busy_until"] - self.t) for c in self.chambers]) / 120,
            np.std([c["busy_until"] for c in self.chambers]) / 120
        ], dtype=np.float32)

    # -----------------------------
    # CHAMBER SELECTION
    # -----------------------------
    def _select_chamber(self, job, action):

        candidates = [
            c for c in self.chambers
            if c["busy_until"] <= self.t and
            max(s["current"] for s in c["slots"]) >= job["required_current"]
        ]

        if not candidates:
            return None

        if action == 0:
            return candidates[0]

        if action == 1:
            same = [c for c in candidates if c["type"] == job["type"]]
            return same[0] if same else candidates[0]

        if action == 2:
            return max(candidates, key=lambda c: max(s["current"] for s in c["slots"]))

        if action == 3:
            return min(candidates, key=lambda c: c["busy_until"])

        return random.choice(candidates)

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):

        self.t += 1

        job = self.queue.pop(0)
        self.total_jobs += 1

        reward = 0.0

        chamber = self._select_chamber(job, action)

        # FAILURE
        if chamber is None:

            job["wait"] += 1
            self.dropped += 1 if job["wait"] > 10 else 0

            if job["wait"] <= 10:
                self.queue.append(job)

            reward -= 1.0

        # SUCCESS
        else:

            chamber["busy_until"] = self.t + job["duration"]
            self.completed += 1

            if chamber["type"] != job["type"]:
                self.reconfigs += 1
                reward += 5.0
            else:
                reward += 10.0

        # SYSTEM METRICS
        busy = sum(c["busy_until"] > self.t for c in self.chambers)

        C = self.completed / max(1, self.total_jobs)
        D = self.dropped / max(1, self.total_jobs)
        U = busy / self.n_ch
        R = self.reconfigs / max(1, self.total_jobs)

        reward += 100*C - 80*D + 50*U - 30*R

        self.queue.append(self._new_job())
        self.total_reward += reward

        done = self.t >= self.max_steps

        return self._obs(), float(reward), done, False, {}

    # -----------------------------
    # METRICS
    # -----------------------------
    def get_utilization(self):
        busy = sum(c["busy_until"] > self.t for c in self.chambers)
        return busy / self.n_ch

    def get_completion_rate(self):
        return self.completed / max(1, self.total_jobs)

    def get_reconfigs(self):
        return self.reconfigs

    def get_total_reward(self):
        return float(self.total_reward)