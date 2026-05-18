import gymnasium as gym
import numpy as np
import json
import random


class ResourceEnv(gym.Env):

    def __init__(
        self,
        data_path,
        queue_size=50000,
        max_steps=20000
    ):

        super().__init__()
        with open(data_path, "r") as f:
            self.chambers = json.load(f)

        self.n_ch = len(self.chambers)

        # =================================================
        # ACTIONS
        # =================================================
      
        self.action_space = gym.spaces.Discrete(6)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(9,),
            dtype=np.float32
        )

        self.queue_size = queue_size
        self.max_steps = max_steps

        self.reset()

    # =====================================================
    # JOB GENERATOR
    # =====================================================

    def _new_job(self):

        return {

            "type": random.choice(
                ["Cycling", "Performance"]
            ),

            "duration": random.randint(
                100,
                500
            ),

            "required_current": random.choice(
                [150, 200, 250, 300, 400, 500]
            ),

            "priority": random.randint(1, 5)
        }

    # =====================================================
    # RESET
    # =====================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.t = 0

        self.completed = 0

        self.reconfigs = 0

        self.waiting = 0

        self.total_jobs = 0

        self.total_reward = 0

        for ch in self.chambers:

            ch["busy_until"] = 0

        self.queue = [

            self._new_job()

            for _ in range(self.queue_size)
        ]

        return self._obs(), {}

    # =====================================================
    # OBSERVATION
    # =====================================================

    def _obs(self):

        busy = sum(

            c["busy_until"] > self.t

            for c in self.chambers
        )

        utilization = busy / self.n_ch

        job = self.queue[0]

        load_variance = np.std([

            max(
                0,
                c["busy_until"] - self.t
            )

            for c in self.chambers

        ]) / 500

        return np.array([

            utilization,

            len(self.queue) / self.queue_size,

            job["duration"] / 500,

            job["required_current"] / 500,

            job["priority"] / 5,

            self.waiting / max(
                1,
                self.total_jobs
            ),

            self.reconfigs / max(
                1,
                self.total_jobs
            ),

            self.completed / max(
                1,
                self.total_jobs
            ),

            load_variance

        ], dtype=np.float32)

    # =====================================================
    # CHAMBER SELECTION
    # =====================================================

    def _select_chamber(
        self,
        job,
        strategy
    ):

        candidates = []

        for ch in self.chambers:

            # skip busy chambers
            if ch["busy_until"] > self.t:
                continue

            compatible = False

            max_current = 0

            for sl in ch["slots"]:

                max_current = max(
                    max_current,
                    sl["current"]
                )

                if sl["current"] >= job["required_current"]:

                    compatible = True

            if not compatible:
                continue

            score = 0

            # =============================================
            # SAME TYPE
            # =============================================

            if strategy == 0:

                if ch["type"] == job["type"]:

                    score += 100

            # =============================================
            # MAX UTILIZATION
            # =============================================

            elif strategy == 1:

                score += (
                    1000 - ch["busy_until"]
                )

            # =============================================
            # MIN RECONFIG
            # =============================================

            elif strategy == 2:

                if ch["type"] == job["type"]:

                    score += 200

            # =============================================
            # HIGHEST CURRENT
            # =============================================

            elif strategy == 3:

                score += max_current

            # =============================================
            # BALANCED
            # =============================================

            elif strategy == 4:

                if ch["type"] == job["type"]:

                    score += 50

                score += (
                    max_current * 0.5
                )

            # =============================================
            # RANDOM
            # =============================================

            elif strategy == 5:

                score += random.random()

            candidates.append(
                (score, ch)
            )

        if len(candidates) == 0:

            return None

        candidates.sort(

            key=lambda x: x[0],

            reverse=True
        )

        return candidates[0][1]

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):

        self.t += 1

        reward = 0

        job = self.queue.pop(0)

        self.total_jobs += 1

        chamber = self._select_chamber(
            job,
            action
        )

        # =================================================
        # NO CHAMBER FOUND
        # =================================================

        if chamber is None:

            self.waiting += 1

            self.queue.append(job)

            reward -= 10

        else:

            chamber["busy_until"] = (

                self.t + job["duration"]
            )

            self.completed += 1

            reward += 15

            # =============================================
            # RECONFIG
            # =============================================

            if chamber["type"] != job["type"]:

                self.reconfigs += 1

                reward -= 5

            else:

                reward += 5

        # =================================================
        # METRICS
        # =================================================

        busy = sum(

            c["busy_until"] > self.t

            for c in self.chambers
        )

        utilization = busy / self.n_ch

        completion = (

            self.completed
            / max(1, self.total_jobs)
        )

        reconfig = (

            self.reconfigs
            / max(1, self.total_jobs)
        )

        waiting = (

            self.waiting
            / max(1, self.total_jobs)
        )

        # =================================================
        # FINAL REWARD
        # =================================================

        reward += (

            50 * utilization +

            30 * completion -

            15 * reconfig -

            12 * waiting
        )

        # IMPORTANT
        self.total_reward += reward

        # add next job
        self.queue.append(
            self._new_job()
        )

        done = self.t >= self.max_steps

        return (

            self._obs(),

            float(reward),

            done,

            False,

            {}
        )

    # =====================================================
    # METRICS
    # =====================================================

    def get_utilization(self):

        busy = sum(

            c["busy_until"] > self.t

            for c in self.chambers
        )

        return busy / self.n_ch

    def get_completion_rate(self):

        return (

            self.completed
            / max(1, self.total_jobs)
        )

    def get_reconfigs(self):

        return self.reconfigs

    def get_total_reward(self):

        return self.total_reward