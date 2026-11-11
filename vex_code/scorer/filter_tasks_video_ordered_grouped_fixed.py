import json


# ==========================================================
# Config
# ==========================================================


backup_result = __import__("os").environ.get("VEX_BACKUP_RESULT", 'data/test_results.json')
padding_result = __import__("os").environ.get("VEX_PADDING_RESULT", 'data/test_results.json')
verifier_result = __import__("os").environ.get("VEX_VERIFIER_RESULT", 'data/test_results.json')
testfile = __import__("os").environ.get("VEX_TESTFILE", 'data/testset_full.json')
description_file = __import__("os").environ.get("VEX_DESCRIPTION_FILE", 'data/vs2_desc_fromSFT.json')


nsample = 1
threshold = 7
max_trial = 3


# ==========================================================
# Normalize task id
# ==========================================================

def normalize_task_id(task_id):
    """
    Normalize task IDs so that:
        xxx_verified
        xxx_video_0
        xxx_sample_0
        xxx_verified_video_0
        xxx_verified_sample_0

    are all mapped to:
        xxx
    """

    task_id = str(task_id)

    # remove video/sample suffix
    task_id = task_id.split("_video_")[0]
    task_id = task_id.split("_sample")[0]

    # remove trailing "_verified"
    if task_id.endswith("_verified"):
        task_id = task_id[:-len("_verified")]

    return task_id


def load_verifier_results(path):
    """Read JSON arrays or JSONL and normalize verifier field names."""
    with open(path) as fin:
        text = fin.read()
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}: invalid JSONL at line {line_number}") from exc
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON array or JSONL records")
    normalized = []
    skipped = 0
    for index, record in enumerate(records):
        if record.get("error"):
            skipped += 1
            continue
        item = dict(record)
        if "task_id" not in item and "task" in item:
            item["task_id"] = item["task"]
        if "ref" not in item and "result" in item:
            item["ref"] = item["result"]
        if "verification" in item:
            item["pred"] = item["verification"]["score"]
            item["ref"] = item["conversations"][-1]["value"]
        for key in ("task_id", "pred", "ref"):
            if key not in item:
                raise ValueError(f"{path}: record {index} missing {key}")
        normalized.append(item)
    print(f"Loaded {len(normalized)} verifier records from {path}; skipped errors={skipped}")
    return normalized


# ==========================================================
# Read descriptions
# ==========================================================

with open(description_file) as fin:
    data = json.load(fin)

task_to_lengths = {}

for task, desc in data.items():

    task = normalize_task_id(task)

    task_to_lengths[task] = len(desc.split())

    if "I can not get useful knowledge" in desc:
        task_to_lengths[task] = 1


# ==========================================================
# Read trajectory finish status
# ==========================================================

finished_map = {}

with open(
    'data/traj_data_video_vs2_qwen_harness_traj_aggo_cuagym.json'
) as fin:
    trajdata = json.load(fin)


for datapiece in trajdata:

    task_id = normalize_task_id(datapiece["task"])

    action = "".join(
        datapiece["steps"][-1]["action"].split("<tool_call>")[1:]
    )

    if "terminate" not in action:
        finished_map[task_id] = False
    else:
        finished_map[task_id] = True

# ==========================================================
# Read original task set
# ==========================================================

with open(testfile) as fin:
    origtasks = json.load(fin)


# ==========================================================
# Build done_tasks
#
# structure:
#
# done_tasks[task_id] = [
#     [prediction, reference, priority],
#     ...
# ]
# ==========================================================

done_tasks = {}


# ----------------------------------------------------------
# Backup results
# ----------------------------------------------------------

backup_scores = load_verifier_results(backup_result)

for datapiece in backup_scores:

    if "qwen35" not in datapiece.get("video", ""):

        task_id = normalize_task_id(datapiece["task_id"])

        if task_id not in done_tasks:
            done_tasks[task_id] = []

        if "gemini" in backup_result:
            prediction = float(
                datapiece["verification"]["score"]
            )
            reference = float(
                datapiece["conversations"][-1]["value"]
            )
        else:
            prediction = float(datapiece["pred"])
            reference = float(datapiece["ref"])

        # Backup gets highest priority
        done_tasks[task_id].append(
            [prediction, reference, 100000]
        )


# ----------------------------------------------------------
# Main verifier results
# ----------------------------------------------------------

scores = load_verifier_results(verifier_result)


for datapiece in scores:

    task_id = normalize_task_id(datapiece["task_id"])

    desc_len = task_to_lengths.get(task_id, 1)

    if task_id not in done_tasks:
        done_tasks[task_id] = []

    if "gemini" in verifier_result:
        prediction = float(
            datapiece["verification"]["score"]
        )
        reference = float(
            datapiece["conversations"][-1]["value"]
        )
    else:
        prediction = float(datapiece["pred"])
        reference = float(datapiece["ref"])

    done_tasks[task_id].append(
        [prediction, reference, desc_len]
    )


# ----------------------------------------------------------
# Padding results
# ----------------------------------------------------------

if padding_result != "":

    padding_scores = load_verifier_results(padding_result)

    for datapiece in padding_scores:

        task_id = normalize_task_id(datapiece["task_id"])

        if task_id not in done_tasks:
            done_tasks[task_id] = []

        # NOTE:
        # Here we check padding_result, NOT padding_scores
        if "gemini" in padding_result:
            prediction = float(
                datapiece["verification"]["score"]
            )
            reference = float(
                datapiece["conversations"][-1]["value"]
            )
        else:
            prediction = float(datapiece["pred"])
            reference = float(datapiece["ref"])

        if len(done_tasks[task_id]) < max_trial:
            done_tasks[task_id].append(
                [prediction, reference, 0]
            )


# ==========================================================
# Category mapping
# ==========================================================

CATEGORY_GROUPS = {
    "os": "OS",

    "libreoffice_calc": "Office",
    "libreoffice_impress_verified": "Office",
    "libreoffice_impress": "Office",
    "libreoffice_writer": "Office",

    "chrome": "Daily",
    "vlc": "Daily",
    "thunderbird": "Daily",

    "vscode": "Professional",
    "vs_code": "Professional",
    "vs-code": "Professional",
    "gimp": "Professional",

    "multi_apps": "Workflow",
    "pdf": "PDF",
}


group_stats = {
    group: {
        "total_score": 0.0,
        "total_tasks": 0,
        "total_executions": 0,
    }
    for group in [
        "OS",
        "Office",
        "Daily",
        "Professional",
        "Workflow",
        "PDF"
    ]
}


total_score = 0.0
total_tasks = 0
count = 0

missing_tasks = []


# ==========================================================
# Evaluate
# ==========================================================

for cat, tasks in origtasks.items():

    group = CATEGORY_GROUPS.get(cat)

    if group is None:
        print(f"[Warning] Unknown category: {cat}")
        continue

    for raw_task in tasks:

        # --------------------------------------------------
        # Normalize original task ID
        # --------------------------------------------------

        task = normalize_task_id(raw_task)

        # --------------------------------------------------
        # Check whether result exists
        # --------------------------------------------------

        if task not in done_tasks:
            print(
                f"[Warning] Missing task: "
                f"{raw_task} -> normalized: {task}"
            )

            missing_tasks.append(
                (cat, raw_task, task)
            )

            continue

        if len(done_tasks[task]) == 0:
            print(
                f"[Warning] Empty result list: {task}"
            )
            continue

        # --------------------------------------------------
        # Sort candidates by priority
        # --------------------------------------------------

        all_tasks = sorted(
            done_tasks[task],
            key=lambda tup: tup[2],
            reverse=True
        )

        # --------------------------------------------------
        # Only allow at most max_trial executions
        # --------------------------------------------------

        attempted = all_tasks[:max_trial]

        cursor = 0
        current_score = attempted[0][0]

        while (
            current_score < threshold
            and cursor < len(attempted) - 1
        ):
            cursor += 1
            current_score = attempted[cursor][0]

        # --------------------------------------------------
        # Determine final reference score and executions
        # --------------------------------------------------

        if current_score >= threshold:

            # Successfully reached threshold
            final_score = attempted[cursor][1]
            num_executions = cursor + 1

        else:

            # All available attempts failed threshold.
            # Select the attempted trajectory with
            # highest verifier score.
            best_attempt = max(
                attempted,
                key=lambda tup: tup[0]
            )

            final_score = best_attempt[1]
            num_executions = len(attempted)

        # --------------------------------------------------
        # Overall statistics
        # --------------------------------------------------

        total_score += final_score
        total_tasks += 1
        count += num_executions

        # --------------------------------------------------
        # Group statistics
        # --------------------------------------------------

        group_stats[group]["total_score"] += final_score
        group_stats[group]["total_tasks"] += 1
        group_stats[group]["total_executions"] += num_executions


# ==========================================================
# Overall results
# ==========================================================

print("=" * 60)
print("Overall")
print("=" * 60)

print(f"total tasks: {total_tasks}")

if total_tasks > 0:
    print(
        f"total executions per task: "
        f"{count / total_tasks:.4f}"
    )

    print(
        f"average score: "
        f"{total_score / total_tasks:.4f}"
    )
else:
    print("total executions per task: N/A")
    print("average score: N/A")


# ==========================================================
# Grouped results
# ==========================================================

print("\n" + "=" * 60)
print("Grouped by task categories")
print("=" * 60)


for group in [
    "OS",
    "Office",
    "Daily",
    "Professional",
    "Workflow",
    "PDF"
]:

    stats = group_stats[group]

    n_tasks = stats["total_tasks"]

    if n_tasks == 0:

        print(
            f"{group:15s} | "
            f"tasks: {0:4d} | "
            f"exec/task: N/A | "
            f"avg score: N/A"
        )

        continue

    avg_score = (
        stats["total_score"] / n_tasks
    )

    avg_exec = (
        stats["total_executions"] / n_tasks
    )

    print(
        f"{group:15s} | "
        f"tasks: {n_tasks:4d} | "
        f"exec/task: {avg_exec:.4f} | "
        f"avg score: {avg_score:.4f}"
    )


# ==========================================================
# Missing tasks
# ==========================================================

print("\n" + "=" * 60)
print("Missing tasks")
print("=" * 60)

print(f"number of missing tasks: {len(missing_tasks)}")

for cat, raw_task, normalized_task in missing_tasks:
    print(
        f"{cat:30s} | "
        f"{raw_task} -> {normalized_task}"
    )