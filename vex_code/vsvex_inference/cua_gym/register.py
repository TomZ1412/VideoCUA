"""Register the CUA-Gym python-reward getter and metric on OSWorld evaluators.

The dance:

* ``DesktopEnv._set_evaluator_info`` resolves names by attribute lookup:

  - ``result_getter = getattr(getters, "get_{type}")``
  - ``metric        = getattr(metrics, func)``

* So all we need to do is install two attributes on those modules.

Both functions are tolerant: failures degrade to ``REWARD: 0.0`` rather than
crashing the rollout, so a flaky VM or a malformed reward.py won't poison the
whole training batch.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("cua_gym.register")

_REWARD_RE = re.compile(r"REWARD:\s*([0-9]*\.?[0-9]+)")


def get_cua_gym_python_reward(env, config: Dict[str, Any]) -> str:
    """Upload ``reward.py`` to the VM, run it, return captured stdout (string).

    OSWorld evaluator getter contract::

        getters.get_<type>(env, evaluator["result"]) -> result_state

    The ``result_state`` we return is the raw stdout text. The metric below
    parses out the ``REWARD: x.xx`` line.
    """
    local_path = config.get("local_path")
    remote_dir = config.get("remote_dir", "/tmp")
    task_id = config.get("task_id") or "task"

    if not local_path or not os.path.isfile(local_path):
        logger.error("cua_gym reward script missing locally: %s", local_path)
        return ""

    sc = env.setup_controller
    suffix = uuid.uuid4().hex[:8]
    short_id = str(task_id)[:8].replace("/", "_")
    remote_path = f"{remote_dir.rstrip('/')}/cua_gym_reward_{short_id}_{suffix}.py"
    stdout_file = f"cua_gym_reward_{short_id}_{suffix}.stdout.txt"
    stderr_file = f"cua_gym_reward_{short_id}_{suffix}.stderr.txt"

    # 1. Upload reward.py to the VM via OSWorld's existing file upload endpoint.
    try:
        sc._upload_file_setup([{"local_path": str(local_path), "path": remote_path}])
    except Exception as e:
        logger.exception("cua_gym: failed to upload reward.py for task %s: %s", task_id, e)
        return ""

    # 2. Execute it; OSWorld's _execute_setup writes stdout/stderr files into
    # ``cache_dir`` (already set per-task by DesktopEnv._set_task_info).
    try:
        sc._execute_setup(
            command=["python3", remote_path],
            stdout=stdout_file,
            stderr=stderr_file,
            shell=False,
        )
    except Exception as e:
        logger.exception("cua_gym: reward.py execution raised for task %s: %s", task_id, e)
        # Fall through; partial stdout may still have been written.

    out_path = Path(sc.cache_dir) / stdout_file
    err_path = Path(sc.cache_dir) / stderr_file
    stdout = out_path.read_text(encoding="utf-8", errors="replace") if out_path.exists() else ""
    stderr = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""

    if stderr.strip():
        # Truncate to keep logs manageable.
        logger.info(
            "cua_gym reward.py stderr (task=%s, tail):\n%s",
            task_id, stderr[-2000:],
        )
    if not stdout.strip():
        logger.warning("cua_gym reward.py produced no stdout (task=%s)", task_id)

    return stdout


def cua_gym_python_reward(stdout: str, **opts: Any) -> float:
    """Parse the last ``REWARD: x.xx`` line from reward.py stdout. Default 0.0.

    OSWorld metric contract::

        metric(result_state, **options) -> float

    We accept any ``**opts`` so callers can pass extras (e.g. for clamping)
    without breaking the signature.
    """
    if not stdout:
        return 0.0
    matches = _REWARD_RE.findall(str(stdout))
    if not matches:
        logger.warning("cua_gym: no REWARD: line found in reward stdout (len=%d)", len(stdout))
        return 0.0
    try:
        score = float(matches[-1])
    except ValueError:
        return 0.0
    # Clamp to [0, 1] defensively; CUA-Gym rubrics are designed within that range.
    return max(0.0, min(1.0, score))


def register_cua_gym_evaluators() -> None:
    """Install the cua_gym getter + metric on OSWorld's evaluator namespaces.

    Idempotent. Call once per process before invoking ``DesktopEnv.reset()``
    with a CUA-Gym-derived ``task_config``.
    """
    # Lazy import so importing cua_gym alone never requires OSWorld to be installed.
    from desktop_env.evaluators import getters, metrics

    setattr(getters, "get_cua_gym_python_reward", get_cua_gym_python_reward)
    setattr(metrics, "cua_gym_python_reward", cua_gym_python_reward)

    logger.info(
        "cua_gym evaluators registered: "
        "getters.get_cua_gym_python_reward, metrics.cua_gym_python_reward"
    )
