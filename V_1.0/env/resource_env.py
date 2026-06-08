import gymnasium as gym
import numpy as np
import json
import random


class ResourceEnv(gym.Env):

    def __init__(self, data_path, max_steps=3000):

        super().__init__()

        with open(data_path, "r") as f:
            self.chambers = json.load(f)

        self.n_ch = len(self.chambers)

        # PPO chooses strategy (NOT chamber ID)
        self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(10,),
            dtype=np.float32
        )

        self.max_steps = max_steps
        self.reset()

    # =====================================================
    # JOB GENERATION (REALISTIC LOAD)
    # =====================================================
    def _new_job(self):
        return {
            "type": random.choice(["Cycling", "Performance"]),
            "duration": random.randint(10, 50),
            "required_current": random.choice([150, 200, 250, 300, 400]),
            "priority": random.randint(1, 5),
            "wait": 0
        }

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.t = 0
        self.completed = 0
        self.reconfigs = 0
        self.dropped = 0
        self.total_jobs = 0
        self.total_reward = 0.0

        for ch in self.chambers:
            ch["busy_until"] = 0

        self.queue = [self._new_job() for _ in range(40)]

        return self._obs(), {}

    # =====================================================
    # OBSERVATION
    # =====================================================
    def _obs(self):

        busy = sum(c["busy_until"] > self.t for c in self.chambers)

        job = self.queue[0]

        return np.array([
            busy / self.n_ch,
            len(self.queue) / 40,
            job["duration"] / 50,
            job["required_current"] / 400,
            job["priority"] / 5,
            self.completed / max(1, self.total_jobs),
            self.dropped / max(1, self.total_jobs),
            self.reconfigs / max(1, self.total_jobs),
            np.mean([max(0, c["busy_until"] - self.t) for c in self.chambers]) / 50,
            np.std([c["busy_until"] for c in self.chambers]) / 50
        ], dtype=np.float32)

    # =====================================================
    # CHAMBER SELECTION STRATEGY
    # =====================================================
    def _select_chamber(self, job, strategy):

        candidates = []

        for ch in self.chambers:

            if ch["busy_until"] > self.t:
                continue

            max_current = max(s["current"] for s in ch["slots"])

            if max_current < job["required_current"]:
                continue

            candidates.append(ch)

        if not candidates:
            return None

        if strategy == 0:
            return candidates[0]

        if strategy == 1:
            same = [c for c in candidates if c["type"] == job["type"]]
            return same[0] if same else candidates[0]

        if strategy == 2:
            return max(candidates, key=lambda c: max(s["current"] for s in c["slots"]))

        if strategy == 3:
            return min(candidates, key=lambda c: c["busy_until"])

        return random.choice(candidates)

    # =====================================================
    # STEP FUNCTION (FIXED + STABLE)
    # =====================================================
    def step(self, action):

        self.t += 1

        job = self.queue.pop(0)
        self.total_jobs += 1

        reward = 0.0

        chamber = self._select_chamber(job, action)

        # -------------------------------------------------
        # NO VALID CHAMBER
        # -------------------------------------------------
        if chamber is None:

            job["wait"] += 1

            if job["wait"] > 10:
                self.dropped += 1
            else:
                self.queue.append(job)

            reward -= 1.0

        # -------------------------------------------------
        # SUCCESSFUL ASSIGNMENT
        # -------------------------------------------------
        else:

            chamber["busy_until"] = self.t + job["duration"]
            self.completed += 1

            if chamber["type"] == job["type"]:
                reward += 10.0
            else:
                reward += 6.0
                self.reconfigs += 1

        # -------------------------------------------------
        # SYSTEM METRICS
        # -------------------------------------------------
        busy = sum(c["busy_until"] > self.t for c in self.chambers)

        utilization = busy / self.n_ch
        completion = self.completed / max(1, self.total_jobs)
        drop_rate = self.dropped / max(1, self.total_jobs)
        reconfig_rate = self.reconfigs / max(1, self.total_jobs)

        # =================================================
        # YOUR FORMULA (FIXED + STABLE)
        # R = αC – βD – γ(1-U) – δR(C)
        # =================================================

        alpha = 120.0
        beta = 100.0
        gamma = 60.0
        delta = 30.0

        reward += (
            alpha * completion
            - beta * drop_rate
            - gamma * (1.0 - utilization)
            - delta * reconfig_rate
        )

        self.queue.append(self._new_job())

        self.total_reward += reward

        done = self.t >= self.max_steps

        return self._obs(), float(reward), done, False, {}

    # =====================================================
    # METRICS
    # =====================================================
    def get_utilization(self):
        busy = sum(c["busy_until"] > self.t for c in self.chambers)
        return busy / self.n_ch

    def get_completion_rate(self):
        return self.completed / max(1, self.total_jobs)

    def get_reconfigs(self):
        return self.reconfigs

    def get_total_reward(self):
        return self.total_reward