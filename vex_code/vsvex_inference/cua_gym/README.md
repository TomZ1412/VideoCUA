# `cua_gym/` — CUA-Gym task adapter for OSWorld-verified-AIIC

Plug CUA-Gym dataset bundles (7,897 tasks at v1) into the existing OSWorld
`DesktopEnv` / `VolcengineProvider` stack with **zero changes to OSWorld core
code**.

This package adds:

| Public symbol | What it does |
|---|---|
| `load_cua_gym_task(task_dir, ...)` | Reads a CUA-Gym task bundle (`task.json`, `initial_setup.py`, `reward.py`) and returns an OSWorld-compatible `task_config` dict you can pass directly to `DesktopEnv.reset(task_config=...)`. |
| `register_cua_gym_evaluators()` | Installs a new getter (`get_cua_gym_python_reward`) and metric (`cua_gym_python_reward`) on the OSWorld `evaluators.getters` / `evaluators.metrics` namespaces. Call once per process. |

## Why this is needed

CUA-Gym `task.json` differs from OSWorld's native task config in two places:

| Field | OSWorld | CUA-Gym | Adapter does |
|---|---|---|---|
| `config[].type == "download"` | `parameters.files[].url` is an **HTTP(S) URL** that OSWorld fetches via `requests.get` then uploads. | A relative path like `"./initial_setup.py"`. | Rewrites the step into OSWorld's already-existing `upload_file` step with `local_path` resolved against the staged task dir. |
| `evaluator` | `{ "func": "...", "result": {"type": "...", ...}, "expected": ..., "options": ... }` | `{ "type": "python", "url": "./reward.py" }` | Rewrites the dict so `func=cua_gym_python_reward` and `result.type=cua_gym_python_reward`; the registered getter uploads + executes the reward script and the registered metric parses the `REWARD: x.xx` line. |

The original OSWorld code paths (`SetupController._upload_file_setup`,
`SetupController._execute_setup`, the evaluators dispatch in `DesktopEnv`) are
reused as-is.

## Architecture

### Option A — All-in-one VM image (recommended)

The CUA-Gym mocks are baked into the OSWorld VM image. Each task VM
starts with mocks already running on `localhost:8000-8095`. This is the
simplest deployment and the one used for the reference smoke test below.

```
  ┌─────────── Host (RL trainer / agent process) ───────────────┐
  │                                                              │
  │  from cua_gym import load_cua_gym_task, register_cua_gym_evaluators │
  │  task = load_cua_gym_task(task_dir, env_file=".env.cua-gym") │
  │  env  = DesktopEnv(provider_name="volcengine")               │
  │  env.reset(task_config=task)                                 │
  │                                                              │
  └──────────────┬──────────────────────────────────────────────┘
                 │  SetupController HTTP   (via PROXY_IP:3128)
  ┌──────────────▼──────────────────────────────────────────────┐
  │ OSWorld VM  —  YOUR_IMAGE_ID                   │
  │  :5000   OSWorld backend                                     │
  │  :8000-8095  98 CUA-Gym mock apps  (all localhost)          │
  │                                                              │
  │  initial_setup.py → calls http://localhost:808x/post         │
  │  Chrome → opens  http://localhost:808x/?sid=<uuid>           │
  │  reward.py      → calls http://localhost:808x/go             │
  │  prints  REWARD: x.xx                                        │
  └─────────────────────────────────────────────────────────────┘
```

`.env.cua-gym` values for this setup all use `http://127.0.0.1:<port>`.

### Option B — Separate mock server VM

The OSWorld VM and the mock server are two separate VMs in the same VPC
subnet. Use this when you want the mock state to persist across many
training episodes without re-injecting it.

```
  ┌──────────────────────┐          ┌──────────────────────────┐
  │  OSWorld VM          │  private │  Mock Server VM          │
  │  (per-task, fresh)   │──subnet──▶  (long-running)          │
  │  initial_setup.py    │          │  :8000-8097  98 mocks    │
  │  reward.py           │          └──────────────────────────┘
  └──────────────────────┘
```

`.env.cua-gym` values use the mock VM's **private IP**
(e.g. `http://192.168.1.5:8047`).

## Quick start

```python
from desktop_env.desktop_env import DesktopEnv
from cua_gym import load_cua_gym_task, register_cua_gym_evaluators

# register AFTER importing DesktopEnv (so OSWorld evaluator modules are loaded first)
env = DesktopEnv(provider_name="volcengine")
register_cua_gym_evaluators()

task = load_cua_gym_task(
    "/data/cua_gym_tasks/0018392e-cfed-5e2f-8278-a4580d64a00a",
    mock_url_env_file="/data/.env.cua-gym",
    url_variables_manifest="/data/CUA-Gym-data/url_variables.json",
)

obs = env.reset(task_config=task)
# ... agent rollout via env.step(action) ...
score = env.evaluate()    # uploads reward.py to VM, runs, parses REWARD: x.xx
```

End-to-end CLI: `python cua_gym/examples/run_one_task.py --task-dir <dir>
--env-file <path> --manifest <path>`.

## Smoke test runbook

Complete steps to verify the integration on a fresh server.

### 1. Prerequisites

| Item | Value |
|---|---|
| OSWorld VM image (with mocks baked in) | `YOUR_IMAGE_ID` |
| Mocks available inside VM at | `localhost:8000-8095` |
| OSWorld backend inside VM at | `localhost:5000` |
| Host → VM proxy | `PROXY_IP=163.7.4.140` (port 3128, user `osworld`) |
| Dataset archive | `CUA-Gym-data/artifacts/cua_gym_tasks_v1.tar.zst` |
| Manifest | `CUA-Gym-data/url_variables.json` |

### 2. Extract the task dataset (once)

```bash
mkdir -p /tmp/cua_gym_tasks
zstd -d CUA-Gym-data/artifacts/cua_gym_tasks_v1.tar.zst --stdout \
    | tar xf - -C /tmp/cua_gym_tasks
# -> 7897 task directories under /tmp/cua_gym_tasks/
```

### 3. Generate `.env.cua-gym` (once per image/port layout)

Mocks run on `localhost` **inside** the VM, so use `127.0.0.1`:

```bash
python OSWorld-verified-AIIC/cua_gym/examples/gen_env_file.py \
    --mock-vm-ip 127.0.0.1 \
    --manifest CUA-Gym-data/url_variables.json \
    --output .env.cua-gym
# -> writes 32 CUA_GYM_*_URL variables
```

### 4. Create `.env` with Volcengine credentials

```bash
# OSWorld-verified-AIIC/.env  (already committed as .env.example)
VOLCENGINE_ACCESS_KEY_ID=<your key>
VOLCENGINE_SECRET_ACCESS_KEY=<your secret>
VOLCENGINE_REGION=ap-southeast-3
VOLCENGINE_IMAGE_ID=YOUR_IMAGE_ID
VOLCENGINE_INSTANCE_TYPE=["ecs.g3al.2xlarge","ecs.g3il.2xlarge","ecs.g3a.2xlarge"]
VOLCENGINE_SECURITY_GROUP_ID=<sg id>
VOLCENGINE_SUBNET_ID=<subnet id>
VOLCENGINE_ZONE_ID=ap-southeast-3a
VOLCENGINE_DEFAULT_PASSWORD=<vm password>
PROXY_IP=163.7.4.140
```

### 5. Install dependencies

```bash
# Inside your Python env (must have OSWorld-verified-AIIC requirements):
pip install -r OSWorld-verified-AIIC/requirements.txt
```

### 6. Run the smoke test (no agent, expect REWARD: 0.0)

```bash
cd OSWorld-verified-AIIC
python cua_gym/examples/run_one_task.py \
    --task-dir /tmp/cua_gym_tasks/0018392e-cfed-5e2f-8278-a4580d64a00a \
    --env-file ../.env.cua-gym \
    --manifest ../CUA-Gym-data/url_variables.json \
    --provider volcengine \
    --no-agent \
    -vv
```

Expected output:
```
======================================================================
Task ID:      0018392e-cfed-5e2f-8278-a4580d64a00a
App:          instagram_mock
Difficulty:   hard
Instruction:  I'm preparing for a collaboration announcement...
Config steps: 2
Evaluator:    func=cua_gym_python_reward result.type=cua_gym_python_reward
======================================================================
>> Resetting environment (provisioning VM, running initial_setup.py)...
>> reset complete.
>> --no-agent set, skipping rollout phase.
>> Evaluating (uploading reward.py, running, parsing)...
======================================================================
RESULT: score = 0.0
======================================================================
```

A score of `0.0` is correct — no agent performed the task. The important
thing is that the pipeline completed without an exception.

### 7. Verify individual stages independently

```bash
# Stage 1: loader dry-run (no VM needed)
python -c "
import sys; sys.path.insert(0, '.')
from cua_gym.loader import load_cua_gym_task
task = load_cua_gym_task(
    '/tmp/cua_gym_tasks/0018392e-cfed-5e2f-8278-a4580d64a00a',
    mock_url_env_file='../.env.cua-gym',
    url_variables_manifest='../CUA-Gym-data/url_variables.json',
)
import json; s = json.dumps(task['config'])
assert '__CUA_GYM_' not in s, 'residual placeholder!'
print('OK - no placeholders, task ID:', task['id'])
"

# Stage 2: check staged files have correct URLs
grep BASE_URL /tmp/cua_gym_stage/0018392e-*/initial_setup.py
# -> BASE_URL = 'http://127.0.0.1:8047'
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `EnvironmentError: VOLCENGINE_IMAGE_ID must be set` | `.env` not loaded | Ensure `.env` is in `OSWorld-verified-AIIC/` or export vars manually |
| `RuntimeError: Missing CUA-Gym endpoint env vars` | `.env.cua-gym` missing/wrong path | Re-run `gen_env_file.py` and check `--output` path |
| `reward.py` timeout / connection refused | Mock not running on the VM | Verify image; SSH in and `curl http://localhost:8047` |
| `REWARD:` line missing in stdout | Task pre-condition not met | Normal for `--no-agent` runs on tasks that require state interaction |
| `ModuleNotFoundError: No module named 'volcenginesdkcore'` | SDK not installed | `pip install "volcengine-python-sdk[ecs,vpc]" -i https://pypi.org/simple/` |

## Network / deployment prerequisites

### All-in-one image (Option A — recommended)

One VM does everything. The only network path that matters:

| From | To | Goes via | What it carries |
|---|---|---|---|
| Host (RL trainer) | VM private IP | `PROXY_IP=163.7.4.140` (port 3128) — already wired into `SetupController` / `PythonController` | screenshots, /setup/upload, /setup/execute, pyautogui actions |

`initial_setup.py` / `reward.py` hit `http://localhost:808x` **inside**
the same VM — no cross-VM traffic at all.

`.env.cua-gym` must use `127.0.0.1` (not the external proxy IP).

### Separate mock VM (Option B)

| From | To | Goes via | What it carries |
|---|---|---|---|
| Host | OSWorld VM | `PROXY_IP:3128` | control plane |
| OSWorld VM | Mock VM private IP | **VPC private subnet, direct** | `initial_setup.py` / `reward.py` HTTP calls |

`.env.cua-gym` must use the mock VM's **private IP**
(e.g. `http://192.168.1.5:8047`), never the public/proxy IP.

### Generate `.env.cua-gym` with one command

The `examples/gen_env_file.py` helper expands the manifest into a complete env
file targeting your mock VM. It also comments out variables for any mocks that
failed health-checks (default: Expensify, PACS-viewer, TradingView — none of
which have any tasks referencing them in the v1 dataset, so this is purely
cosmetic).

    python cua_gym/examples/gen_env_file.py \
        --mock-vm-ip 10.0.0.5 \
        --manifest /data/CUA-Gym-data/url_variables.json \
        --output /data/.env.cua-gym

    # Custom base-port (if you ran deploy-all.sh with a different one):
    python cua_gym/examples/gen_env_file.py --mock-vm-ip 10.0.0.5 --base-port 18000 ...

    # Custom port map (TSV: <port>\t<mock_dir> per line, e.g. from healthcheck.sh):
    python cua_gym/examples/gen_env_file.py --mock-vm-ip 10.0.0.5 \
        --port-map /tmp/port-map.tsv ...

    # Override failed-mock list:
    python cua_gym/examples/gen_env_file.py --mock-vm-ip 10.0.0.5 \
        --failed-mocks expensify_mock,pacs-viewer_mock ...

`load_cua_gym_task` reads the resulting `.env.cua-gym` file and replaces
`__CUA_GYM_*__` placeholders inside the staged copy of `initial_setup.py`
and `reward.py`. The original dataset on disk is **never modified**.

## VM image requirements

The OSWorld VM image must have:

- Python 3.8+ (already required by OSWorld)
- `requests` (already used by OSWorld's own backend server)
- Chrome (already required by OSWorld)

CUA-Gym `libreoffice_*` / `vscode` / `pdf` tasks additionally need:

- `openpyxl`, `python-docx`, `python-pptx`, `pdfplumber`, etc., depending on
  the task family. The standard OSWorld qcow2 image ships these.

If your image is a stripped-down derivative, pass `ensure_requests=True` to
`load_cua_gym_task` to prepend a `pip install requests` step. For other
packages, install them into your image once rather than per-task.

## Filtering tasks by `app_type`

`load_cua_gym_task` preserves `app_type` and `difficulty` on the returned
dict, so you can filter before constructing the env::

    import json
    from pathlib import Path

    web_only = []
    for d in Path("/data/cua_gym_tasks").iterdir():
        if not (d / "task.json").is_file():
            continue
        meta = json.loads((d / "task.json").read_text())
        if (meta.get("app_type") or "").endswith("_mock"):
            web_only.append(d)

Or load metadata from the parquet table at `CUA-Gym-data/data/tasks.parquet`
(see the dataset README) for vectorized filtering.

## Failure modes & how the adapter handles them

| Symptom | Cause | Adapter behavior |
|---|---|---|
| `RuntimeError: Missing CUA-Gym endpoint env vars` | env file / shell env doesn't cover a `__CUA_GYM_*__` placeholder used by the task. | Loader fails fast at `load_cua_gym_task` time, before any VM is provisioned. Pass `use_hosted_defaults=True` only for smoke tests. |
| `RuntimeError: files still contain unreplaced __CUA_GYM_* placeholders` | env file provided but a needed key is missing. | Loader fails fast. |
| `reward.py` prints traceback before `REWARD:` | mock unreachable, malformed state, etc. | Getter still returns full stdout (stderr is logged separately). Metric returns `0.0` if no `REWARD:` line is found. |
| `reward.py` prints multiple `REWARD:` lines | Some tasks include intermediate component scores. | Metric uses the **last** match. |
| Network glitch uploading reward.py | OSWorld's `_upload_file_setup` retries 3× with backoff. | If all retries fail, getter logs and returns `""`, score becomes `0.0`. |
| `evaluator.type` is not `"python"` | Future / mixed dataset versions | Loader passes the evaluator dict through unchanged (back-compat with OSWorld native format). |

## Concurrency notes

* Each task uses a **fresh VM** (Volcengine provider does that via
  `revert_to_snapshot` → `delete + create from image`), so `/tmp/task_web_sid`
  inside the VM is per-task by construction. No cross-episode contamination.
* The mock server **is shared** across all concurrent OSWorld VMs. Isolation
  is provided by the per-task `sid` UUID that `initial_setup.py` generates and
  writes to `/tmp/task_web_sid`. Reward script reads the same file inside the
  same VM.
* The staging cache `$TMPDIR/cua_gym_stage/<task_id>/` is keyed by task id, so
  re-loading the same task from multiple threads is safe. Loading **different**
  tasks concurrently is safe (each one creates its own subdir).

## Cleanup

* The staging cache survives across runs. Wipe with `rm -rf $TMPDIR/cua_gym_stage`
  whenever you re-materialize against new mock URLs.
* On the mock server side, run the daily cron / `cleanup.sh` shipped with
  `CUA-Gym-Hub/deploy/volcano/` to purge old `.mock-states/` and
  `.mock-files/` entries.

## Limitations

* The adapter assumes the CUA-Gym v1 task layout (`task.json` + sibling
  scripts). Future CUA-Gym versions that change `task.json` schema (e.g.
  add new step types) will fall through unchanged; only `download` and
  `evaluator.type == "python"` are translated.
* `initial_setup.py` writes the per-task sid to `/tmp/task_web_sid` inside the
  VM. If you stack multiple CUA-Gym tasks in the same VM without reverting
  the snapshot, only the most recent task's sid survives. The default
  `DesktopEnv` behaviour (revert on `reset()`) is correct here.
* For tasks where `app_type` is `multi_apps` and the rubric needs cross-mock
  state, the mock server VM must host all referenced mocks. The default
  `bootstrap.sh` deployment hosts all 98, so this just works.
