import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm


def get_grouped_category_name(category):
    """Maps app-level categories to high-level task groups."""
    normalized = category.lower()
    if "multi_apps" in normalized or "multi_app" in normalized or "workflow" in normalized:
        return "Workflow"
    if "libreoffice_calc" in normalized or "libreoffice_impress" in normalized or "libreoffice_writer" in normalized:
        return "Office"
    if normalized.startswith("chrome") or "thunderbird" in normalized or normalized.startswith("vlc"):
        return "Daily"
    if "vs_code" in normalized or "vscode" in normalized or "gimp" in normalized:
        return "Professional"
    return "OS"


def build_grouped_stats(stats):
    grouped_stats = defaultdict(lambda: {"actual_tasks": 0, "theoretical_tasks": 0, "total_avg_score": 0.0})
    for category, data in stats.items():
        grouped_category = get_grouped_category_name(category)
        grouped_stats[grouped_category]["actual_tasks"] += data.get("actual_tasks", 0)
        grouped_stats[grouped_category]["theoretical_tasks"] += data.get("theoretical_tasks", 0)
        grouped_stats[grouped_category]["total_avg_score"] += data.get("total_avg_score", 0.0)
    return grouped_stats


def get_grouped_category_name(category):
    """Maps app-level categories to high-level task groups."""
    normalized = category.lower()
    if normalized in {"libreoffice_calc", "libreoffice_impress", "libreoffice_writer"}:
        return "Office"
    if normalized in {"chrome", "vlc", "vlc_player", "thunderbird"}:
        return "Daily"
    if normalized in {"vs_code", "vscode", "gimp"}:
        return "Professional"
    if normalized in {"multi_apps", "multi_app", "workflow"}:
        return "Workflow"
    return "OS"


def build_grouped_stats(stats):
    grouped_stats = defaultdict(lambda: {"actual_tasks": 0, "theoretical_tasks": 0, "total_avg_score": 0.0})
    for category, data in stats.items():
        grouped_category = get_grouped_category_name(category)
        grouped_stats[grouped_category]["actual_tasks"] += data.get("actual_tasks", 0)
        grouped_stats[grouped_category]["theoretical_tasks"] += data.get("theoretical_tasks", 0)
        grouped_stats[grouped_category]["total_avg_score"] += data.get("total_avg_score", 0.0)
    return grouped_stats


def get_task_list(task_file_path):
    """从 JSON 里读取 category -> task list。"""
    try:
        with open(task_file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Task file not found at {task_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {task_file_path}")
        return None


def discover_tasks_from_results(results_dir):
    """
    如果不提供 task_file，则直接从 results_dir 自动发现任务。
    约定：
    - task/
    - task_sample_1/
    - task_sample_2/
    ...
    会被归并为同一个 task。
    """
    tasks_by_cat = {}

    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found at {results_dir}")
        return None

    for category in sorted(os.listdir(results_dir)):
        category_dir = os.path.join(results_dir, category)
        if not os.path.isdir(category_dir):
            continue

        task_ids = set()
        for entry in os.listdir(category_dir):
            entry_path = os.path.join(category_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            if "_sample_" in entry:
                base_task_id = entry.rsplit("_sample_", 1)[0]
            elif "_audio_" in entry:
                base_task_id = entry.rsplit("_audio_", 1)[0]
            else:
                base_task_id = entry

            task_ids.add(base_task_id)

        if task_ids:
            tasks_by_cat[category] = sorted(task_ids)

    return tasks_by_cat


def load_tasks(task_file, results_dir):
    if task_file == "auto":
        print("Task file is set to auto, discovering tasks from results directory...")
        return discover_tasks_from_results(results_dir)
    return get_task_list(task_file)


def get_score(file_path):
    """读取 result.txt 中的分数。"""
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            return float(content)
    except (FileNotFoundError, ValueError, TypeError):
        return 0.0


def is_done(traj_path):
    """
    判断 rollout 是否正常完成。
    与第二个脚本一致：检查 traj.jsonl 最后一条 action 是否包含 terminate。
    同时兼容旧格式中的 DONE。
    """
    return True
    try:
        with open(traj_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return False

        last_line = lines[-1]

        # 兼容旧格式
        if "DONE" in last_line:
            return True

        # 与第二个代码一致
        if "<tool_call>" in last_line:
            action_str = "".join(last_line.split("<tool_call>")[1:])
        else:
            action_str = last_line

        return "terminate" in action_str

    except (FileNotFoundError, OSError):
        return False


def build_task_dir_candidates(task_id, sample_idx):
    """
    兼容两种目录命名：
    - task_id / task_id_sample_N
    - task_id_audio_N

    这里把 sample_indices 视为 0-based rollout 序号：
    - 0 -> task_id 或 task_id_audio_1
    - 1 -> task_id_sample_1 或 task_id_audio_2
    """
    normalized_task_id = task_id.replace("_verified", "")
    candidates = []
    if sample_idx == 0:
        candidates.append(normalized_task_id)
    else:
        candidates.append(f"{normalized_task_id}_sample_{sample_idx}")
    candidates.append(f"{normalized_task_id}_audio_{sample_idx + 1}")
    candidates.append(f"{normalized_task_id}_video_{sample_idx + 1}")
    return list(dict.fromkeys(candidates))


def calculate_topk_mean(results_dir, tasks_by_cat, sample_indices):
    """
    与第二个脚本保持一致的统计逻辑：

    1. 已完成 rollout -> finished_scores
    2. 未完成 rollout -> failed_scores
    3. 如果存在已完成 rollout，只使用已完成 rollout
    4. 如果没有任何已完成 rollout，但存在未完成 rollout，
       fallback 到未完成 rollout 的 score
    5. 如果连 score 都没有，则该 task 不参与统计
    """

    stats = defaultdict(
        lambda: {
            "actual_tasks": 0,
            "total_avg_score": 0.0,
        }
    )

    unfinished = {}
    task_to_score = {}
    task_details = {}

    for category, task_list in tqdm(
        tasks_by_cat.items(),
        desc="Categories"
    ):
        stats[category]["theoretical_tasks"] = len(task_list)
        task_to_score[category] = {}
        task_details[category] = {}

        for task_id in tqdm(
            task_list,
            desc=category,
            leave=False
        ):
            finished_scores = []
            failed_scores = []

            missing_samples = []
            unfinished_samples = []
            sample_records = []

            for sample_idx in sample_indices:

                candidate_task_dirs = build_task_dir_candidates(
                    task_id,
                    sample_idx
                )

                task_dir_name = candidate_task_dirs[0]

                result_path = os.path.join(
                    results_dir,
                    category,
                    task_dir_name,
                    "result.txt"
                )

                traj_path = os.path.join(
                    results_dir,
                    category,
                    task_dir_name,
                    "traj.jsonl"
                )

                # Find actually existing candidate
                for candidate_task_dir in candidate_task_dirs:

                    candidate_result_path = os.path.join(
                        results_dir,
                        category,
                        candidate_task_dir,
                        "result.txt"
                    )

                    if os.path.exists(candidate_result_path):

                        task_dir_name = candidate_task_dir
                        result_path = candidate_result_path

                        traj_path = os.path.join(
                            results_dir,
                            category,
                            candidate_task_dir,
                            "traj.jsonl"
                        )

                        break

                result_exists = os.path.exists(result_path)
                done = is_done(traj_path)

                record = {
                    "sample_idx": sample_idx,
                    "task_dir": task_dir_name,
                    "candidate_task_dirs": candidate_task_dirs,
                    "result_path": result_path,
                    "traj_path": traj_path,
                    "exists": result_exists,
                    "done": done,
                    "score": None,
                }

                # --------------------------------------------------
                # Same logic as second script
                # --------------------------------------------------

                if result_exists:
                    score = get_score(result_path)
                    record["score"] = score

                    if done:
                        # Finished rollout
                        finished_scores.append(score)

                    else:
                        # Unfinished rollout
                        failed_scores.append(score)
                        unfinished_samples.append(sample_idx)

                else:
                    # No result file at all
                    missing_samples.append(sample_idx)

                sample_records.append(record)

            # --------------------------------------------------
            # Same fallback logic as second script
            # --------------------------------------------------

            used_scores = finished_scores
            fallback_to_failed = False

            if len(failed_scores) > 0 and len(finished_scores) == 0:
                used_scores = failed_scores
                fallback_to_failed = True

            # --------------------------------------------------
            # Record task details
            # --------------------------------------------------

            task_details[category][task_id] = {
                "sample_indices": sample_indices,
                "sample_records": sample_records,
                "missing_samples": missing_samples,
                "unfinished_samples": unfinished_samples,
                "finished_scores": finished_scores,
                "unfinished_scores": failed_scores,
                "fallback_to_unfinished": fallback_to_failed,
                "used_score_count": len(used_scores),
            }

            # --------------------------------------------------
            # No available score at all
            # --------------------------------------------------

            if len(used_scores) == 0:

                if category not in unfinished:
                    unfinished[category] = {}

                unfinished[category][task_id] = {
                    "missing_samples": missing_samples,
                    "unfinished_samples": unfinished_samples,
                    "reason": "no_available_scores",
                }

                continue

            # --------------------------------------------------
            # Mean mode, same as second script
            # --------------------------------------------------

            avg_score = sum(used_scores) / len(used_scores)

            task_to_score[category][task_id] = avg_score
            task_details[category][task_id]["avg_score"] = avg_score

            stats[category]["actual_tasks"] += 1
            stats[category]["total_avg_score"] += avg_score

    return (
        stats,
        unfinished,
        task_to_score,
        task_details,
    )


def print_results_table(stats, sample_indices):
    """格式化打印统计结果。"""
    title = f"Mean of Samples {sample_indices}"

    try:
        from prettytable import PrettyTable
    except ImportError:
        print("\nWarning: `prettytable` is not installed. Falling back to basic table format.")
        print("To install, run: pip install prettytable")
        print_results_table_basic(stats, title)
        return

    table = PrettyTable()
    table.title = title
    table.field_names = [
        "Category",
        "Total Score",
        "Actual Tasks",
        "Theoretical",
        "Completion",
        "SR (Avg Score)",
    ]
    table.align["Category"] = "l"
    for field in table.field_names[1:]:
        table.align[field] = "r"

    total_actual = 0
    total_theoretical = 0
    total_avg_score = 0.0

    for category in sorted(stats.keys()):
        data = stats[category]
        actual = data["actual_tasks"]
        theoretical = data.get("theoretical_tasks", 0)
        total_score = data["total_avg_score"]

        total_actual += actual
        total_theoretical += theoretical
        total_avg_score += total_score

        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"

        table.add_row([
            category,
            f"{total_score:.2f}",
            actual,
            theoretical,
            completion_rate,
            avg_score,
        ])

    table.add_row(["-" * 25, "-" * 12, "-" * 12, "-" * 12, "-" * 12, "-" * 15], divider=True)

    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_avg_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"

    table.add_row([
        "Overall",
        f"{total_avg_score:.2f}",
        total_actual,
        total_theoretical,
        overall_completion,
        overall_avg_score,
    ])

    print(table)

    grouped_stats = build_grouped_stats(stats)
    grouped_table = PrettyTable()
    grouped_table.title = "Grouped by Task Categories"
    grouped_table.field_names = ["Group", "Total Score", "Actual Tasks", "Theoretical", "Completion", "SR (Avg Score)"]
    grouped_table.align["Group"] = "l"
    for field in grouped_table.field_names[1:]:
        grouped_table.align[field] = "r"

    total_actual = 0
    total_theoretical = 0
    total_avg_score = 0.0
    group_order = ["OS", "Office", "Daily", "Professional", "Workflow"]

    for group in group_order:
        if group not in grouped_stats:
            continue
        data = grouped_stats[group]
        actual = data["actual_tasks"]
        theoretical = data.get("theoretical_tasks", 0)
        total_score = data["total_avg_score"]
        total_actual += actual
        total_theoretical += theoretical
        total_avg_score += total_score
        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"
        grouped_table.add_row([group, f"{total_score:.2f}", actual, theoretical, completion_rate, avg_score])

    grouped_table.add_row(["-" * 25, "-" * 12, "-" * 12, "-" * 12, "-" * 12, "-" * 15], divider=True)
    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_avg_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"
    grouped_table.add_row(["Overall", f"{total_avg_score:.2f}", total_actual, total_theoretical, overall_completion, overall_avg_score])
    print(grouped_table)


def print_results_table_basic(stats, title):
    """基础格式打印。"""
    print(f"\n--------- {title} ---------")
    header = (
        f"| {'Category':<25} | {'Total Score':>12} | {'Actual Tasks':>12} | "
        f"{'Theoretical':>12} | {'Completion':>12} | {'SR (Avg Score)':>15} |"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_actual = 0
    total_theoretical = 0
    total_avg_score = 0.0

    for category in sorted(stats.keys()):
        data = stats[category]
        actual = data["actual_tasks"]
        theoretical = data.get("theoretical_tasks", 0)
        total_score = data["total_avg_score"]

        total_actual += actual
        total_theoretical += theoretical
        total_avg_score += total_score

        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"

        print(
            f"| {category:<25} | {total_score:>12.2f} | {actual:>12} | "
            f"{theoretical:>12} | {completion_rate:>12} | {avg_score:>15} |"
        )

    print("-" * len(header))

    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_avg_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"

    print(
        f"| {'Overall':<25} | {total_avg_score:>12.2f} | {total_actual:>12} | "
        f"{total_theoretical:>12} | {overall_completion:>12} | {overall_avg_score:>15} |"
    )
    print("-" * len(header))

    grouped_stats = build_grouped_stats(stats)
    print("\n--------- Grouped by Task Categories ---------")
    grouped_header = (
        f"| {'Group':<25} | {'Total Score':>12} | {'Actual Tasks':>12} | "
        f"{'Theoretical':>12} | {'Completion':>12} | {'SR (Avg Score)':>15} |"
    )
    print("-" * len(grouped_header))
    print(grouped_header)
    print("-" * len(grouped_header))

    total_actual = 0
    total_theoretical = 0
    total_avg_score = 0.0
    for group in ["OS", "Office", "Daily", "Professional", "Workflow"]:
        if group not in grouped_stats:
            continue
        data = grouped_stats[group]
        actual = data["actual_tasks"]
        theoretical = data.get("theoretical_tasks", 0)
        total_score = data["total_avg_score"]
        total_actual += actual
        total_theoretical += theoretical
        total_avg_score += total_score
        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"
        print(
            f"| {group:<25} | {total_score:>12.2f} | {actual:>12} | "
            f"{theoretical:>12} | {completion_rate:>12} | {avg_score:>15} |"
        )

    print("-" * len(grouped_header))
    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_avg_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"
    print(
        f"| {'Overall':<25} | {total_avg_score:>12.2f} | {total_actual:>12} | "
        f"{total_theoretical:>12} | {overall_completion:>12} | {overall_avg_score:>15} |"
    )
    print("-" * len(grouped_header))


def main():
    parser = argparse.ArgumentParser(
        description="Calculate task scores by averaging the first selected samples."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/vs2_agentnet_osworld_trainproj_8samples",
        help="结果目录。",
    )
    parser.add_argument(
        "--task_file",
        type=str,
        default="auto",
        help="任务列表 JSON；设为 auto 时从 results_dir 自动发现任务。",
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="要取平均的 rollout 下标，默认前 3 个：[0, 1, 2]；兼容 task_sample_N 与 task_audio_N。",
    )
    # parser.add_argument(
    #     "--allow_partial",
    #     action="store_true",
    #     help="若设置，则允许在部分 sample 缺失时对已有 sample 求平均。",
    # )

    args = parser.parse_args()

    sample_indices = sorted(set(args.sample_indices))
    tasks_by_cat = load_tasks(args.task_file, args.results_dir)
    if not tasks_by_cat:
        return

    stats, unfinished, task_to_score, task_details = calculate_topk_mean(
        args.results_dir,
        tasks_by_cat,
        sample_indices=sample_indices,
        # require_all_samples=not args.allow_partial,
    )

    print_results_table(stats, sample_indices)

    suffix = "_".join(str(x) for x in sample_indices)
    unfinished_file = os.path.join(args.results_dir, f"unfinished_topavg_{suffix}.json")
    score_file = os.path.join(args.results_dir, f"task_to_score_topavg_{suffix}.json")
    details_file = os.path.join(args.results_dir, f"task_details_topavg_{suffix}.json")

    with open(unfinished_file, "w") as f:
        json.dump(unfinished, f, indent=4, ensure_ascii=False)

    with open(score_file, "w") as f:
        json.dump(task_to_score, f, indent=4, ensure_ascii=False)

    with open(details_file, "w") as f:
        json.dump(task_details, f, indent=4, ensure_ascii=False)

    print(f"\nSaved unfinished tasks to: {unfinished_file}")
    print(f"Saved task scores to: {score_file}")
    print(f"Saved task details to: {details_file}")


if __name__ == "__main__":
    main()