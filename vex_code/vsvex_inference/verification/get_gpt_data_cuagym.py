#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from cua_gym import load_cua_gym_task
CUA_GYM_TASK_ROOT = Path('YOUR_CUAGYM_TASK_ROOT_PATH')


def normalize_task_name(name: str) -> str:
    name = name.replace("_verified", "")
    name = re.sub(r"_sample\d+$", "", name)
    name = re.sub(r"_video_\d+$", "", name)
    return name


def load_whitelist(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_result_index(basepath: Path):
    exact_index = {}
    normalized_index = defaultdict(list)

    for category_dir in sorted(basepath.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for task_dir in sorted(category_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            exact_index[(category, task_dir.name)] = task_dir
            normalized_index[(category, normalize_task_name(task_dir.name))].append(task_dir)

    return exact_index, normalized_index


def resolve_task_dirs(category, task, exact_index, normalized_index):
    clean_task = task.replace("_verified", "")
    exact_match = exact_index.get((category, clean_task))
    if exact_match is not None:
        return [exact_match]

    normalized_key = (category, normalize_task_name(clean_task))
    return list(normalized_index.get(normalized_key, []))


def find_init_screenshot(task_path: Path):
    matches = sorted(task_path.glob("step_0*"))
    return str(matches[0]) if matches else ""


def read_result_score(result_path: Path):
    with result_path.open("r", encoding="utf-8") as f:
        return float(f.readline().strip())


def load_task_config(examples_root: Path, category: str, task_name: str):
    # task_base = normalize_task_name(task_name)
    # config_path = examples_root / category / f"{task_base}.json"
    # with config_path.open("r", encoding="utf-8") as f:
    #     return json.load(f)
    task_dir = CUA_GYM_TASK_ROOT / task_name.rsplit("_video_", 1)[0].rsplit("_sample", 1)[0]
    single_task_config = load_cua_gym_task(
        str(task_dir),
        mock_url_env_file=None,
        url_variables_manifest=None,
        use_hosted_defaults=False,
        ensure_requests=False,
    )
    return single_task_config


def load_steps(task_path: Path):
    trajpath = task_path / "traj.jsonl"
    with trajpath.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    cursor = len(lines) - 1
    while cursor > 0 and json.loads(lines[cursor])["step_num"] > 1:
        cursor -= 1
    lines = lines[cursor:]

    steps = []
    for line in lines:
        step = json.loads(line)
        response = step["response"].split("Action: ")[-1].split("\n<tool_call>")[0].strip().lstrip()
        steps.append(
            {
                "description": response,
                "action": step["response"],
                "screenshot": str(task_path / step["screenshot_file"]),
            }
        )
    return steps


def build_piece(category, task_dir: Path, task_config, model: str, sample_id: int):
    result_path = task_dir / "result.txt"
    init_screenshot = find_init_screenshot(task_dir)
    if not init_screenshot:
        raise FileNotFoundError(f"missing step_0*: {task_dir}")

    return {
        "category": category,
        "task": task_dir.name if sample_id < 0 else f"{task_dir.name}_sample{sample_id}",
        "instruction": task_config["instruction"],
        "init_screenshot": init_screenshot,
        "result": read_result_score(result_path),
        "model": model,
        "sample_id": sample_id,
        "steps": load_steps(task_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build trajectory JSON from result directories with robust whitelist-to-directory matching."
    )
    parser.add_argument(
        "--whitelist",
        type=Path,
        default=Path('YOUR_TASK_MANIFEST_PATH'),
    )
    parser.add_argument(
        "--basepath",
        type=Path,
        default=Path('YOUR_BENCHMARK_RESULTS_PATH'),
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=Path('YOUR_TASK_CONFIG_ROOT_PATH'),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path('YOUR_TRAJECTORIES_OUTPUT_PATH'),
    )
    parser.add_argument("--model", default="vs2")
    parser.add_argument("--nsample", type=int, default=1)
    args = parser.parse_args()

    whitelist = load_whitelist(args.whitelist)
    exact_index, normalized_index = build_result_index(args.basepath)

    newdata = []
    processed_dirs = set()
    stats = defaultdict(int)
    reason_examples = defaultdict(list)

    for category, tasks in whitelist.items():
        for task in tqdm(tasks, desc=f"category={category}"):
            for sample_id in range(args.nsample):
                candidate_dirs = resolve_task_dirs(category, task, exact_index, normalized_index)
                if not candidate_dirs:
                    stats["missing_task_dir"] += 1
                    if len(reason_examples["missing_task_dir"]) < 20:
                        reason_examples["missing_task_dir"].append(f"{category}/{task}")
                    continue

                for task_dir in candidate_dirs:
                    unique_key = (category, task_dir.name, sample_id)
                    if unique_key in processed_dirs:
                        stats["deduplicated_dir"] += 1
                        continue

                    result_path = task_dir / "result.txt"
                    if not result_path.exists():
                        stats["missing_result_txt"] += 1
                        if len(reason_examples["missing_result_txt"]) < 20:
                            reason_examples["missing_result_txt"].append(f"{category}/{task_dir.name}")
                        continue

                    try:
                        task_config = load_task_config(args.examples_root, category, task_dir.name)
                        piece = build_piece(category, task_dir, task_config, args.model, sample_id)
                    except Exception as e:
                        stats["build_piece_failed"] += 1
                        if len(reason_examples["build_piece_failed"]) < 20:
                            reason_examples["build_piece_failed"].append(f"{category}/{task_dir.name}: {e}")
                        continue

                    processed_dirs.add(unique_key)
                    newdata.append(piece)
                    stats["accepted"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        json.dump(newdata, fout, ensure_ascii=False, indent=4)

    unique_tasks = {
        (item["category"], normalize_task_name(item["task"]))
        for item in newdata
    }

    print(f"written: {args.output}")
    print(f"accepted entries: {len(newdata)}")
    print(f"accepted unique tasks: {len(unique_tasks)}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")

    for reason, examples in sorted(reason_examples.items()):
        print(f"\n[{reason}]")
        for example in examples:
            print(example)


if __name__ == "__main__":
    main()