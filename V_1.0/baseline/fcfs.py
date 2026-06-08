import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from env.resource_env import ResourceEnv


# =========================================================
# FCFS + FIRST-FIT BASELINE
# =========================================================

def run_fcfs(seed=42):

    env = ResourceEnv("data/chambers.json")
    obs, _ = env.reset(seed=seed)

    done = False

    while not done:

        env.t += 1
        reward = 0

        # FCFS: first job in queue
        job = env.queue.pop(0)
        env.total_jobs += 1

        assigned = False

        # First-fit assignment
        for chamber in env.chambers:

            if chamber["busy_until"] > env.t:
                continue

            compatible = False

            for slot in chamber["slots"]:
                if slot["current"] >= job["required_current"]:
                    compatible = True
                    break

            if not compatible:
                continue

            # Assign job
            chamber["busy_until"] = env.t + job["duration"]
            env.completed += 1

            reward += 10

            # Reconfiguration handling (kept, but no penalty)
            if chamber["type"] != job["type"]:
                env.reconfigs += 1
            else:
                reward += 5

            assigned = True
            break

        # If not assigned
        if not assigned:
            env.waiting += 1
            env.queue.append(job)

        # Metrics
        busy = sum(c["busy_until"] > env.t for c in env.chambers)
        utilization = busy / env.n_ch

        completion = env.completed / max(1, env.total_jobs)
        reconfig = env.reconfigs / max(1, env.total_jobs)

        # ONLY positive reward terms (no penalties)
        reward += (
            120 * utilization +
            20 * completion
        )

        env.total_reward += reward

        # New arrival
        env.queue.append(env._new_job())

        done = env.t >= env.max_steps

    return {
        "method": "FCFS",
        "reward": env.get_total_reward(),
        "utilization": env.get_utilization(),
        "completion": env.get_completion_rate(),
        "reconfigs": env.get_reconfigs()
    }