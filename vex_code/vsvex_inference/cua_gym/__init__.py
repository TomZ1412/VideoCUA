"""CUA-Gym task adapter for OSWorld-verified-AIIC.

Zero-touch integration: load a CUA-Gym task bundle, get an OSWorld-compatible
``task_config`` dict, and run it through any provider (volcengine, docker, aws...).

Quick start
-----------

    from desktop_env.desktop_env import DesktopEnv
    from cua_gym import load_cua_gym_task, register_cua_gym_evaluators

    register_cua_gym_evaluators()           # call once per process

    env = DesktopEnv(provider_name="volcengine", os_type="Ubuntu")
    task = load_cua_gym_task(
        "/data/cua_gym_tasks/0018392e-cfed-5e2f-8278-a4580d64a00a",
        mock_url_env_file="/data/.env.cua-gym",
        url_variables_manifest="/data/CUA-Gym-data/url_variables.json",
    )

    obs = env.reset(task_config=task)
    # ... agent rollout ...
    score = env.evaluate()                  # runs reward.py inside the VM
    print("score:", score)

See cua_gym/README.md for the full integration story.
"""

from cua_gym.loader import load_cua_gym_task
from cua_gym.register import (
    cua_gym_python_reward,
    get_cua_gym_python_reward,
    register_cua_gym_evaluators,
)

__all__ = [
    "load_cua_gym_task",
    "register_cua_gym_evaluators",
    "get_cua_gym_python_reward",
    "cua_gym_python_reward",
]
