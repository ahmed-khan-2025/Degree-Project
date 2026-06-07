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

    env = ResourceEnv(
        "data/chambers.json"
    )

    obs, _ = env.reset(seed=seed)

    done = False

    while not done:

        env.t += 1

        reward = 0

        # =====================================================
        # FIRST JOB IN QUEUE (FCFS)
        # =====================================================

        job = env.queue.pop(0)

        env.total_jobs += 1

        assigned = False

        # =====================================================
        # FIRST AVAILABLE COMPATIBLE CHAMBER (FIRST-FIT)
        # =====================================================

        for chamber in env.chambers:

            # chamber busy
            if chamber["busy_until"] > env.t:
                continue

            compatible = False

            for slot in chamber["slots"]:

                if slot["current"] >= job["required_current"]:

                    compatible = True
                    break

            if not compatible:
                continue

            # =================================================
            # ASSIGN JOB
            # =================================================

            chamber["busy_until"] = (
                env.t + job["duration"]
            )

            env.completed += 1

            reward += 10

            # =================================================
            # RECONFIGURATION
            # =================================================

            if chamber["type"] != job["type"]:

                env.reconfigs += 1

                reward -= 5

            else:

                reward += 5

            assigned = True

            break

        # =====================================================
        # NO AVAILABLE CHAMBER
        # =====================================================

        if not assigned:

            env.waiting += 1

            env.queue.append(job)

            reward -= 15

        # =====================================================
        # METRICS
        # =====================================================

        busy = sum(

            c["busy_until"] > env.t

            for c in env.chambers

        )

        utilization = busy / env.n_ch

        completion = (

            env.completed
            / max(1, env.total_jobs)
        )

        reconfig = (

            env.reconfigs
            / max(1, env.total_jobs)
        )

        waiting = (

            env.waiting
            / max(1, env.total_jobs)
        )

        # =====================================================
        # SAME REWARD AS PPO ENVIRONMENT
        # =====================================================

        reward += (

            120 * utilization +
            20 * completion -
            25 * reconfig -
            20 * waiting
        )

        reward -= (

            50 * (1 - utilization)
        )

        env.total_reward += reward

        # =====================================================
        # NEW ARRIVAL
        # =====================================================

        env.queue.append(
            env._new_job()
        )

        done = env.t >= env.max_steps

    return {

        "method": "FCFS",

        "reward": env.get_total_reward(),

        "utilization": env.get_utilization(),

        "completion": env.get_completion_rate(),

        "reconfigs": env.get_reconfigs()
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = run_fcfs()

    print("\n===== FCFS-FIRSTFIT =====")
    print(result)