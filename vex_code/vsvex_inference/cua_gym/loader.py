"""Translate a CUA-Gym task bundle into an OSWorld-compatible ``task_config`` dict.

CUA-Gym dataset bundles have the shape::

    <task_id>/
      task.json          # { id, instruction, app_type, config[], evaluator }
      initial_setup.py   # uploaded to VM and executed during reset()
      reward.py          # uploaded to VM and executed during evaluate()

Two things differ from OSWorld's native task json:

1. ``config[].type == "download"`` uses relative URLs like ``"./initial_setup.py"``.
   OSWorld's ``_download_setup`` expects an HTTP(S) URL (it ``requests.get(...)``
   then multipart-uploads). We rewrite those as the equivalent ``upload_file``
   step which OSWorld already supports out of the box.

2. ``evaluator == {"type": "python", "url": "./reward.py"}`` is a CUA-Gym extension.
   We rewrite it into ``{"func": "cua_gym_python_reward", "result": {"type": ...}}``
   which is then served by the getter / metric registered in ``cua_gym.register``.

The loader does NOT modify the original dataset on disk. It stages a per-task
copy under ``cache_root`` (default: tempdir) and materializes any
``__CUA_GYM_*__`` placeholders there.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("cua_gym.loader")

PLACEHOLDER_PREFIX = "__CUA_GYM_"
PLACEHOLDER_SUFFIX = "__"
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}

DEFAULT_REMOTE_DIR = "/tmp"
DEFAULT_CACHE_ROOT = Path(tempfile.gettempdir()) / "cua_gym_stage"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _build_replacements(
    manifest_path: Optional[Path],
    env_values: Dict[str, str],
    use_defaults: bool,
) -> Dict[str, str]:
    """Build placeholder -> value map. Mirrors CUA-Gym's materialize_dataset_urls.py.

    Unlike the upstream script, this returns whatever can be resolved without
    failing on globally missing variables. Per-task completeness is verified
    later by ``_check_for_residual_placeholders`` so each task only requires
    the variables it actually references.
    """
    if manifest_path is None or not manifest_path.exists():
        # No manifest: map by naming convention CUA_GYM_FOO_URL -> __CUA_GYM_FOO_URL__
        return {
            f"{PLACEHOLDER_PREFIX}{k[len('CUA_GYM_'):]}{PLACEHOLDER_SUFFIX}": v.rstrip("/")
            for k, v in env_values.items()
            if k.startswith("CUA_GYM_") and v
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variables = manifest.get("variables") or {}
    repl: Dict[str, str] = {}

    for placeholder, spec in variables.items():
        env_name = spec["env"]
        value = env_values.get(env_name)
        if value is None and spec.get("kind") == "host" and env_name.endswith("_HOST"):
            url_value = env_values.get(env_name[: -len("_HOST")] + "_URL")
            if url_value:
                value = urlparse(url_value).netloc or url_value
        if value is None and use_defaults:
            value = spec.get("default")
        if value is None:
            continue
        repl[placeholder] = value.rstrip("/") if spec.get("kind") == "url" else value

    return repl


def _materialize_text(text: str, replacements: Dict[str, str]) -> str:
    for placeholder, value in replacements.items():
        if placeholder in text:
            text = text.replace(placeholder, value)
    return text


def _stage_task_files(
    task_dir: Path,
    stage_dir: Path,
    replacements: Optional[Dict[str, str]],
) -> None:
    """Copy task_dir contents into stage_dir; materialize placeholders for text files."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for src in task_dir.iterdir():
        if not src.is_file():
            continue
        dst = stage_dir / src.name
        if replacements and src.suffix in TEXT_SUFFIXES:
            try:
                text = src.read_text(encoding="utf-8")
                dst.write_text(_materialize_text(text, replacements), encoding="utf-8")
                continue
            except UnicodeDecodeError:
                pass
        shutil.copyfile(src, dst)


def _collect_residual_placeholders(stage_dir: Path) -> Dict[str, List[str]]:
    """Return {placeholder_name: [files containing it]} for any unresolved tokens."""
    import re
    pattern = re.compile(re.escape(PLACEHOLDER_PREFIX) + r"[A-Z0-9_]+" + re.escape(PLACEHOLDER_SUFFIX))
    residual: Dict[str, List[str]] = {}
    for src in stage_dir.iterdir():
        if not src.is_file() or src.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = src.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in set(pattern.findall(text)):
            residual.setdefault(token, []).append(src.name)
    return residual


def _resolve_relative(stage_dir: Path, url: str) -> Path:
    """Resolve a relative or file:// url against the staged task directory."""
    if url.startswith("file://"):
        url = url[len("file://"):]
    if Path(url).is_absolute():
        return Path(url)
    return (stage_dir / url).resolve()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_cua_gym_task(
    task_dir: str | os.PathLike,
    *,
    mock_url_env_file: Optional[str | os.PathLike] = None,
    url_variables_manifest: Optional[str | os.PathLike] = None,
    use_hosted_defaults: bool = False,
    materialize: bool = True,
    cache_root: Optional[str | os.PathLike] = None,
    ensure_requests: bool = False,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    raise_on_placeholder: bool = True,
) -> Dict[str, Any]:
    """Load one CUA-Gym task bundle and return an OSWorld-compatible task_config.

    Parameters
    ----------
    task_dir
        Path to an extracted task directory, e.g. ``cua_gym_tasks/<uuid>/``.
    mock_url_env_file
        Optional .env file containing ``CUA_GYM_*_URL=...`` lines. Required when
        ``materialize=True`` and task files still contain ``__CUA_GYM_*__``
        placeholders.
    url_variables_manifest
        Path to ``CUA-Gym-data/url_variables.json``. When omitted, env vars are
        mapped to placeholders purely by naming convention.
    use_hosted_defaults
        Fall back to the manifest's ``xlang.ai`` defaults for missing variables.
        Smoke tests only.
    materialize
        Stage a copy of the task and replace ``__CUA_GYM_*__`` placeholders in it.
        Default True. Disable if you have already materialized the dataset in-place.
    cache_root
        Directory under which per-task staging directories are created. Default
        ``$TMPDIR/cua_gym_stage``.
    ensure_requests
        Prepend a ``pip install requests`` step to ``config[]``. Most OSWorld
        images already have ``requests``; only needed for stripped-down images.
    remote_dir
        Where ``reward.py`` is uploaded inside the VM. Default ``/tmp``.
    raise_on_placeholder
        If True (default), raise when staged files still contain ``__CUA_GYM_*__``
        after materialization (means env vars are missing). Disable to allow
        partially materialized tasks to run.

    Returns
    -------
    dict
        Ready to pass to ``DesktopEnv.reset(task_config=...)``.
    """
    task_dir = Path(task_dir).expanduser().resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task_dir not found: {task_dir}")

    task_json_path = task_dir / "task.json"
    if not task_json_path.is_file():
        raise FileNotFoundError(f"task.json missing under {task_dir}")

    task = json.loads(task_json_path.read_text(encoding="utf-8"))
    task_id = task.get("id") or task_dir.name

    # --- Materialize placeholders into a staging copy ---
    if materialize:
        env_values: Dict[str, str] = {**os.environ}
        if mock_url_env_file:
            env_values.update(_read_env_file(Path(mock_url_env_file).expanduser().resolve()))
        manifest_path = (
            Path(url_variables_manifest).expanduser().resolve()
            if url_variables_manifest else None
        )
        replacements: Optional[Dict[str, str]] = _build_replacements(
            manifest_path, env_values, use_defaults=use_hosted_defaults
        )
    else:
        replacements = None

    stage_root = Path(cache_root).expanduser().resolve() if cache_root else DEFAULT_CACHE_ROOT
    stage_dir = stage_root / task_id
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    _stage_task_files(task_dir, stage_dir, replacements)

    if materialize and raise_on_placeholder:
        residual = _collect_residual_placeholders(stage_dir)
        if residual:
            # __CUA_GYM_FOO_URL__ -> CUA_GYM_FOO_URL (strip leading and trailing "__")
            needed_env_vars = sorted({tok.strip("_") for tok in residual})
            details = "; ".join(f"{tok} in {','.join(files)}" for tok, files in residual.items())
            raise RuntimeError(
                f"task {task_id}: {len(residual)} placeholder(s) left after materialization. "
                f"Set these env vars (in shell or mock_url_env_file): "
                f"{needed_env_vars}. Tokens: {details}"
            )

    # --- Translate config[] ---
    src_config = task.get("config") or []
    new_config: List[Dict[str, Any]] = []

    if ensure_requests:
        new_config.append({
            "type": "execute",
            "parameters": {
                "command": "python3 -m pip install --quiet --disable-pip-version-check requests",
                "shell": True,
            },
        })

    for step in src_config:
        stype = step.get("type")
        params = step.get("parameters") or {}

        if stype == "download":
            upload_files: List[Dict[str, str]] = []
            download_files: List[Dict[str, str]] = []
            for f in (params.get("files") or []):
                url = (f.get("url") or "").strip()
                target = f.get("path")
                if not url or not target:
                    continue
                if url.startswith(("http://", "https://", "ftp://", "ftps://")):
                    download_files.append({"url": url, "path": target})
                else:
                    local = _resolve_relative(stage_dir, url)
                    if not local.exists():
                        raise FileNotFoundError(
                            f"task {task_id}: download.url={url!r} -> {local} "
                            f"does not exist after staging"
                        )
                    upload_files.append({"local_path": str(local), "path": target})
            if upload_files:
                new_config.append({
                    "type": "upload_file",
                    "parameters": {"files": upload_files},
                })
            if download_files:
                new_config.append({
                    "type": "download",
                    "parameters": {"files": download_files},
                })

        elif stype == "execute":
            cmd = params.get("command")
            # If command is a string with spaces, OSWorld requires shell=True.
            shell = params.get("shell")
            if shell is None:
                shell = isinstance(cmd, str) and len(cmd.split()) > 1
            new_params = {"command": cmd, "shell": bool(shell)}
            for opt in ("stdout", "stderr", "until"):
                if opt in params:
                    new_params[opt] = params[opt]
            new_config.append({"type": "execute", "parameters": new_params})

        else:
            new_config.append(step)

    # --- Translate evaluator ---
    src_eval = task.get("evaluator") or {}
    if src_eval.get("type") == "python":
        reward_url = src_eval.get("url") or "./reward.py"
        reward_local = _resolve_relative(stage_dir, reward_url)
        if not reward_local.exists():
            raise FileNotFoundError(
                f"task {task_id}: evaluator.url={reward_url!r} -> {reward_local} "
                f"does not exist after staging"
            )
        new_eval: Dict[str, Any] = {
            "func": "cua_gym_python_reward",
            "result": {
                "type": "cua_gym_python_reward",
                "local_path": str(reward_local),
                "remote_dir": remote_dir,
                "task_id": task_id,
            },
            "options": {},
            # Preserve OSWorld's optional postconfig hook
            "postconfig": src_eval.get("postconfig", []),
        }
    else:
        # Leave OSWorld-native evaluator dicts alone; CUA-Gym current release
        # uses only the python-reward form, but stay forward-compatible.
        new_eval = src_eval

    # --- Assemble final dict ---
    task_config: Dict[str, Any] = {
        "id": task_id,
        "instruction": task.get("instruction", ""),
        "config": new_config,
        "evaluator": new_eval,
        "proxy": False,
    }
    # Preserve task metadata that downstream filtering / logging may want.
    for k in ("app_type", "difficulty", "source", "related_apps", "snapshot"):
        if k in task and task[k] is not None:
            task_config[k] = task[k]

    logger.debug("loaded task %s (app=%s, steps=%d)", task_id, task.get("app_type"), len(new_config))
    return task_config
