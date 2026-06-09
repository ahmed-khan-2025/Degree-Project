import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from env.resource_env import ResourceEnv


def run_fcfs(seed=42):

    env = ResourceEnv("data/chambers.json")
    env.reset(seed=seed)

    while env.t < env.max_steps:

        env.t += 1

        # =========================
        # SAFE QUEUE HANDLING
        # =========================
        if len(env.queue) == 0:
            env.queue.append(env._new_job())

        job = env.queue.pop(0)
        env.total_jobs += 1

        assigned = False

        # =========================
        # FCFS + FIRST FIT
        # =========================
        for ch in env.chambers:

            if ch["busy_until"] > env.t:
                continue

            max_current = max(s["current"] for s in ch["slots"])

            if max_current < job["required_current"]:
                continue

            # assign job
            ch["busy_until"] = env.t + job["duration"]
            env.completed += 1

            # reconfiguration tracking
            if ch["type"] != job["type"]:
                env.reconfigs += 1

            assigned = True
            break

        # =========================
        # if not assigned → requeue
        # =========================
        if not assigned:
            env.dropped += 1
            env.queue.append(job)

        # =========================
        # CONTROLLED ARRIVAL (CRITICAL FOR STABILITY)
        # =========================
        arrival_rate = int(env.n_ch * 0.02)   # 2% load pressure
        arrival_rate = max(10, min(arrival_rate, 60))

        for _ in range(arrival_rate):
            env.queue.append(env._new_job())

    # =========================
    # FINAL METRICS
    # =========================
    busy = sum(c["busy_until"] > env.t for c in env.chambers)

    C = env.completed / max(1, env.total_jobs)
    U = busy / env.n_ch
    R = env.reconfigs / max(1, env.total_jobs)

    reward = 100 * U + 60 * C - 35 * R

    return {
        "run": seed,
        "method": "FCFS",
        "reward": float(reward),
        "utilization": float(U),
        "completion": float(C),
        "reconfigs": int(env.reconfigs)
    }


if __name__ == "__main__":
    print(run_fcfs())