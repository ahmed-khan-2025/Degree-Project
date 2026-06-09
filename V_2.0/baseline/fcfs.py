import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from env.resource_env import ResourceEnv


def run_fcfs(seed=42):

    env = ResourceEnv("data/chambers.json")
    env.reset(seed=seed)

    while env.t < env.max_steps:

        env.t += 1

        job = env.queue.pop(0)
        env.total_jobs += 1

        reward = 0.0
        assigned = False

        for ch in env.chambers:

            if ch["busy_until"] > env.t:
                continue

            if max(s["current"] for s in ch["slots"]) < job["required_current"]:
                continue

            ch["busy_until"] = env.t + job["duration"]
            env.completed += 1

            reward += 10.0

            if ch["type"] != job["type"]:
                env.reconfigs += 1
                reward -= 5.0

            assigned = True
            break

        if not assigned:
            env.queue.append(job)
            reward -= 2.0

        busy = sum(c["busy_until"] > env.t for c in env.chambers)

        C = env.completed / max(1, env.total_jobs)
        U = busy / env.n_ch
        R = env.reconfigs / max(1, env.total_jobs)

        reward += 100*C + 50*U - 30*R

        env.total_reward += reward

        env.queue.append(env._new_job())

    busy = sum(c["busy_until"] > env.t for c in env.chambers)

    return {
        "run": seed,
        "method": "FCFS",
        "reward": float(env.total_reward),
        "utilization": busy / env.n_ch,
        "completion": env.get_completion_rate(),
        "reconfigs": env.get_reconfigs()
    }