import os
import json
from collections import defaultdict
import argparse
from tqdm import tqdm


def get_grouped_category_name(category):
    """Maps app-level categories to high-level task groups using exact category names."""
    normalized = category.lower().strip()

    if normalized == "pdf":
        return "PDF"
    if normalized == "multi_apps":
        return "Workflow"
    if normalized in {"libreoffice_calc", "libreoffice_impress", "libreoffice_writer"}:
        return "Office"
    if normalized in {"chrome", "vlc", "vlc_player", "thunderbird"}:
        return "Daily"
    if normalized in {"vscode", "vs-code", "gimp"}:
        return "Professional"
    if normalized == "multi_apps":
        return "Workflow"
    if normalized == "os":
        return "OS"

    return "OS"


def build_grouped_stats(stats):
    grouped_stats = defaultdict(
        lambda: {"actual_tasks": 0, "theoretical_tasks": 0, "total_max_score": 0.0}
    )
    for category, data in stats.items():
        grouped_category = get_grouped_category_name(category)
        grouped_stats[grouped_category]["actual_tasks"] += data.get("actual_tasks", 0)
        grouped_stats[grouped_category]["theoretical_tasks"] += data.get("theoretical_tasks", 0)
        grouped_stats[grouped_category]["total_max_score"] += data.get("total_max_score", 0.0)
    return grouped_stats


def get_task_list(task_file_path):
    """Loads the task list from the provided JSON file."""
    try:
        with open(task_file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Task file not found at {task_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {task_file_path}")
        return None


def get_score(file_path):
    """Reads the score from a result file."""
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            return float(content)
    except (FileNotFoundError, ValueError, TypeError):
        return 0.0


def calculate_pass_k(base_results_dir, tasks_by_cat, k):
    """Calculates statistics for pass@k."""
    stats = defaultdict(lambda: {"actual_tasks": 0, "total_max_score": 0.0})

    for category, task_list in tasks_by_cat.items():
        stats[category]["theoretical_tasks"] = len(task_list)

        for task_id in task_list:
            rollout_scores = []
            all_rollouts_exist = True
            for i in range(k):
                result_path = os.path.join(base_results_dir, category, task_id, str(i), "result.txt")
                if os.path.exists(result_path):
                    score = get_score(result_path)
                    rollout_scores.append(score)
                else:
                    all_rollouts_exist = False
                    break

            if all_rollouts_exist:
                stats[category]["actual_tasks"] += 1
                if rollout_scores:
                    max_score = max(rollout_scores)
                    stats[category]["total_max_score"] += max_score

    return stats


def calculate_pass(base_results_dir, tasks_by_cat, alt_dir):
    """Calculates statistics for pass@1."""
    stats = defaultdict(lambda: {"actual_tasks": 0, "total_max_score": 0.0})
    unfinished = {}
    wrong = {}

    for category, task_list in tqdm(tasks_by_cat.items()):
        stats[category]["theoretical_tasks"] = len(task_list)
        if category not in wrong:
            wrong[category] = []

        for task_id in tqdm(task_list):
            orig_task_id = task_id
            task_id = task_id.replace("_verified", "")
            rollout_scores = []
            all_rollouts_exist = True
            result_path = os.path.join(base_results_dir, category, task_id, "result.txt")
            if os.path.exists(result_path):
                if alt_dir != "none":
                    traj_path = os.path.join(base_results_dir, category, task_id, "traj.jsonl")
                    alt_path = os.path.join(alt_dir, category, task_id, "result.txt")
                    with open(traj_path) as fin:
                        lines = fin.readlines()
                    if "DONE" not in lines[-1]:
                        if os.path.exists(alt_path):
                            with open(alt_path) as fin:
                                alt_score = get_score(alt_path)
                                rollout_scores.append(alt_score)
                score = get_score(result_path)
                rollout_scores.append(score)
            else:
                if category not in unfinished:
                    unfinished[category] = []
                unfinished[category].append(orig_task_id)
                all_rollouts_exist = False

            if all_rollouts_exist:
                stats[category]["actual_tasks"] += 1
                if rollout_scores:
                    max_score = max(rollout_scores)
                    stats[category]["total_max_score"] += max_score
                    if max_score < 0.5:
                        wrong[category].append(orig_task_id)

    return stats, unfinished, wrong


def print_results_table(stats, k=1):
    """Prints the results in a formatted table."""
    try:
        from prettytable import PrettyTable
    except ImportError:
        print("\nWarning: `prettytable` is not installed. Falling back to basic table format.")
        print("To install, run: pip install prettytable")
        print_results_table_basic(stats, k)
        return

    table = PrettyTable()
    table.title = f"Pass@{k} Results"
    table.field_names = ["Category", "Total Score", "Actual Tasks", "Theoretical", "Completion", "SR (Avg Score)"]
    table.align["Category"] = "l"
    for field in table.field_names[1:]:
        table.align[field] = "r"

    total_actual = 0
    total_theoretical = 0
    total_max_score = 0.0

    sorted_categories = sorted(stats.keys())

    for category in sorted_categories:
        data = stats[category]
        actual = data["actual_tasks"]
        theoretical = data.get("theoretical_tasks", 0)
        total_score = data["total_max_score"]

        total_actual += actual
        total_theoretical += theoretical
        total_max_score += total_score

        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"

        table.add_row([category, f"{total_score:.2f}", actual, theoretical, completion_rate, avg_score])

    table.add_row(["-" * 25, "-" * 12, "-" * 12, "-" * 12, "-" * 12, "-" * 15], divider=True)

    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_max_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"

    table.add_row(["Overall", f"{total_max_score:.2f}", total_actual, total_theoretical, overall_completion, overall_avg_score])
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
    total_max_score = 0.0
    group_order = ["OS", "Office", "Daily", "Professional", "Workflow", "PDF"]

    for group in group_order:
        if group not in grouped_stats:
            continue
        data = grouped_stats[group]
        actual = data["actual_tasks"]
        theoretical = data.get("theoretical_tasks", 0)
        total_score = data["total_max_score"]

        total_actual += actual
        total_theoretical += theoretical
        total_max_score += total_score

        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"
        grouped_table.add_row([group, f"{total_score:.2f}", actual, theoretical, completion_rate, avg_score])

    grouped_table.add_row(["-" * 25, "-" * 12, "-" * 12, "-" * 12, "-" * 12, "-" * 15], divider=True)
    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_max_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"
    grouped_table.add_row(["Overall", f"{total_max_score:.2f}", total_actual, total_theoretical, overall_completion, overall_avg_score])
    print(grouped_table)


def print_results_table_basic(stats, k):
    """Prints the results in a formatted table (basic)."""
    print(f"\n--------- Pass@{k} Results ---------")
    header = f"| {'Category':<25} | {'Total Score':>12} | {'Actual Tasks':>12} | {'Theoretical':>12} | {'Completion':>12} | {'SR (Avg Score)':>15} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_actual = 0
    total_theoretical = 0
    total_max_score = 0.0

    sorted_categories = sorted(stats.keys())

    for category in sorted_categories:
        data = stats[category]
        actual = data["actual_tasks"]
        theoretical = data.get("theoretical_tasks", 0)
        total_score = data["total_max_score"]

        total_actual += actual
        total_theoretical += theoretical
        total_max_score += total_score

        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"

        print(f"| {category:<25} | {total_score:>12.2f} | {actual:>12} | {theoretical:>12} | {completion_rate:>12} | {avg_score:>15} |")

    print("-" * len(header))

    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_max_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"

    print(f"| {'Overall':<25} | {total_max_score:>12.2f} | {total_actual:>12} | {total_theoretical:>12} | {overall_completion:>12} | {overall_avg_score:>15} |")
    print("-" * len(header))

    grouped_stats = build_grouped_stats(stats)
    print("\n--------- Grouped by Task Categories ---------")
    grouped_header = f"| {'Group':<25} | {'Total Score':>12} | {'Actual Tasks':>12} | {'Theoretical':>12} | {'Completion':>12} | {'SR (Avg Score)':>15} |"
    print("-" * len(grouped_header))
    print(grouped_header)
    print("-" * len(grouped_header))

    total_actual = 0
    total_theoretical = 0
    total_max_score = 0.0
    for group in ["OS", "Office", "PDF", "Daily", "Professional", "Workflow"]:
        if group not in grouped_stats:
            continue
        data = grouped_stats[group]
        actual = data["actual_tasks"]
        theoretical = data.get("theoretical_tasks", 0)
        total_score = data["total_max_score"]

        total_actual += actual
        total_theoretical += theoretical
        total_max_score += total_score

        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"
        print(f"| {group:<25} | {total_score:>12.2f} | {actual:>12} | {theoretical:>12} | {completion_rate:>12} | {avg_score:>15} |")

    print("-" * len(grouped_header))
    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_max_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"
    print(f"| {'Overall':<25} | {total_max_score:>12.2f} | {total_actual:>12} | {total_theoretical:>12} | {overall_completion:>12} | {overall_avg_score:>15} |")
    print("-" * len(grouped_header))


def main():
    parser = argparse.ArgumentParser(description="Calculate pass@k success rates for OSWorld tasks.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="os_worlds/gemini_25_evolve/OSWorld-self-evolve/results/task_test_verified/gemini_25_pro/grounding_g3/50/20260227_112644/",
        help="The base directory where task results are stored.",
    )
    parser.add_argument(
        "--task_file",
        type=str,
        default="os_worlds/gemini_25_evolve/OSWorld-self-evolve/OSWorld/evaluation_examples/test_verified.json",
        help="Path to the JSON file containing the list of tasks.",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=1,
        help="Maximum k to calculate for pass@k.",
    )
    parser.add_argument(
        "--alt_dir",
        type=str,
        default="none",
        help="The alternative route directory where task results are stored.",
    )

    args = parser.parse_args()

    tasks_by_cat = get_task_list(args.task_file)
    if not tasks_by_cat:
        return

    stats, unfinished, wrong = calculate_pass(args.results_dir, tasks_by_cat, args.alt_dir)
    print_results_table(stats)

    with open(args.task_file.replace(".json", "_unfinished.json"), "w") as f:
        json.dump(unfinished, f, indent=4)

    with open(args.task_file.replace(".json", "_wrong.json"), "w") as f:
        json.dump(wrong, f, indent=4)


if __name__ == "__main__":
    main()