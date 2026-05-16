import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from env.resource_env import ResourceEnv


def run_fcfs():

    env = ResourceEnv(
        "data/chambers.json"
    )

    obs, _ = env.reset()

    done = False

    while not done:

        # RANDOM / UNCONSTRAINED
        action = 5

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    return {

        "method": "FCFS",

        "reward": env.get_total_reward(),

        "utilization": env.get_utilization(),

        "completion": env.get_completion_rate(),

        "reconfigs": env.get_reconfigs()
    }