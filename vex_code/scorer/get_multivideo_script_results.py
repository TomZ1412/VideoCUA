import os
import json
from collections import defaultdict
import argparse
from tqdm import tqdm

CATEGORY_GROUPS = {
    "os": "OS",

    # Support both naming conventions
    "calc": "Office",
    "impress": "Office",
    "writer": "Office",
    "libreoffice_calc": "Office",
    "libreoffice_impress": "Office",
    "libreoffice_writer": "Office",

    "chrome": "Daily",
    "vlc": "Daily",
    "thunderbird": "Daily",

    "vscode": "Professional",
    "vs-code": "Professional",
    "gimp": "Professional",

    "multi_apps": "Workflow",
    "pdf": "PDF",
}


def get_task_list(task_file_path):
    """Loads the task list from the provided JSON file."""
    try:
        with open(task_file_path, 'r') as f:
            cat_to_tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Task file not found at {task_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {task_file_path}")
        return None
    task_by_cat_to_video = {}
    for category, task_list in cat_to_tasks.items():
        if category not in task_by_cat_to_video:
            task_by_cat_to_video[category] = {}
        for task_id in task_list:
            if "_video_" in task_id:
                real_id = task_id.split("_video_")[0]
            else:
                real_id = task_id
            if real_id not in task_by_cat_to_video[category]:
                task_by_cat_to_video[category][real_id] = []
            task_by_cat_to_video[category][real_id].append(task_id)
    return task_by_cat_to_video

def get_score(file_path):
    """Reads the score from a result file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            return float(content)
    except (FileNotFoundError, ValueError, TypeError):
        return 0.0, incomplete


def calculate_pass(base_results_dir, tasks_by_cat, mode="mean", task_video_to_score=None, bootstrap_dir=None, score_th=0, max_k=100):
    """Calculates statistics for pass@k."""
    stats = defaultdict(lambda: {'actual_tasks': 0, 'total_max_score': 0.0})
    unfinished = {}
    per_task_stats = {}

    for category, task_list in tqdm(tasks_by_cat.items()):
        stats[category]['theoretical_tasks'] = len(task_list)

        for task_id, video_ids in tqdm(task_list.items()):
            rollout_scores = []
            failed_rollout_scores = []
            # high = min(max_k,len(video_ids))
            # low = max(0,high-1)
            # high = len(video_ids)
            # low = 0
            for video_id in video_ids[:max_k]:
                result_path = os.path.join(base_results_dir, category, video_id.replace("_verified", ""), 'result.txt')
                result_traj = os.path.join(base_results_dir, category, video_id.replace("_verified", ""), 'traj.jsonl')
                # if os.path.exists(result_path):
                try:
                    score = get_score(result_path)
                    with open(result_traj, 'r') as fin:
                        actions = fin.readlines()
                    action_str = "".join(actions[-1].split("<tool_call>")[1:])
                    if "_video_" in video_id and "terminate" not in action_str:
                        failed_rollout_scores.append(score)
                    else:
                        if task_video_to_score is not None and video_id in task_video_to_score:
                            if task_video_to_score[video_id] >= score_th:
                                rollout_scores.append((score, task_video_to_score[video_id]))
                        else:
                            rollout_scores.append(score)
                except:
                    if category not in unfinished:
                        unfinished[category] = []
                    unfinished[category].append(video_id)

            # Bootstrap
            if bootstrap_dir != "none" and len(rollout_scores) == 0:
                bootstrap_result = os.path.join(bootstrap_dir, category, task_id.replace("_verified", ""), 'result.txt')
                if os.path.exists(bootstrap_result):
                    score = get_score(bootstrap_result)
                    rollout_scores.append(score)

            per_task_stats[task_id] = {
                "finished_scores": rollout_scores,
                "unfinished_scores": failed_rollout_scores
            }
            if len(failed_rollout_scores) > 0 and len(rollout_scores) == 0 and "_video_" in video_ids[0]:
                rollout_scores = failed_rollout_scores
            if len(rollout_scores) > 0:
                stats[category]['actual_tasks'] += 1
                if isinstance(rollout_scores[0], tuple):
                    rollout_scores = sorted(rollout_scores, key=lambda x: x[1])
                    if mode == "max":
                        stats[category]['total_max_score'] += rollout_scores[-1][0]
                    else:
                        new_rollout = [x[0] for x in rollout_scores]
                        stats[category]['total_max_score'] += sum(new_rollout) / len(new_rollout)
                elif mode == "mean":
                    avg_score = sum(rollout_scores) / len(rollout_scores)
                    stats[category]['total_max_score'] += avg_score
                elif mode == "max":
                    max_score = max(rollout_scores)
                    stats[category]['total_max_score'] += max_score
                elif mode == "min":
                    max_score = min(rollout_scores)
                    stats[category]['total_max_score'] += max_score

    return stats, unfinished, per_task_stats


def aggregate_stats_by_group(stats, category_groups=CATEGORY_GROUPS):
    """Aggregate category-level statistics into high-level groups."""
    grouped_stats = defaultdict(
        lambda: {
            'actual_tasks': 0,
            'theoretical_tasks': 0,
            'total_max_score': 0.0,
        }
    )

    for category, data in stats.items():
        # Keep an unmapped category as its own group instead of silently dropping it.
        group = category_groups.get(category, category)

        grouped_stats[group]['actual_tasks'] += data.get('actual_tasks', 0)
        grouped_stats[group]['theoretical_tasks'] += data.get('theoretical_tasks', 0)
        grouped_stats[group]['total_max_score'] += data.get('total_max_score', 0.0)

    return grouped_stats


def print_results_table(stats, k):
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

    # sorted_categories = sorted(stats.keys())
    sorted_categories = stats.keys()

    for category in sorted_categories:
        data = stats[category]
        actual = data['actual_tasks']
        theoretical = data.get('theoretical_tasks', 0)
        total_score = data['total_max_score']
        
        total_actual += actual
        total_theoretical += theoretical
        total_max_score += total_score

        completion_rate = f"{(actual / theoretical * 100):.2f}%" if theoretical > 0 else "N/A"
        avg_score = f"{(total_score / actual * 100):.2f}%" if actual > 0 else "N/A"

        table.add_row([category, f"{total_score:.2f}", actual, theoretical, completion_rate, avg_score])

    table.add_row(["-"*25, "-"*12, "-"*12, "-"*12, "-"*12, "-"*15], divider=True)

    overall_completion = f"{(total_actual / total_theoretical * 100):.2f}%" if total_theoretical > 0 else "N/A"
    overall_avg_score = f"{(total_max_score / total_actual * 100):.2f}%" if total_actual > 0 else "N/A"
    
    table.add_row(["Overall", f"{total_max_score:.2f}", total_actual, total_theoretical, overall_completion, overall_avg_score])
    
    print(table)

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

    # sorted_categories = sorted(stats.keys())
    sorted_categories = stats.keys()
    # sorted_categories = stats

    for category in sorted_categories:
        data = stats[category]
        actual = data['actual_tasks']
        theoretical = data.get('theoretical_tasks', 0)
        total_score = data['total_max_score']
        
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


def main():
    parser = argparse.ArgumentParser(description="Calculate pass@k success rates for OSWorld tasks.")
    parser.add_argument(
        '--results_dir',
        type=str,
        default="OSWorld-self-evolve/results/task_test_verified/gemini_25_pro/grounding_g3/50/20260227_112644/",
        help="The base directory where task results are stored."
    )
    parser.add_argument(
        '--task_file',
        type=str,
        default="OSWorld-self-evolve/videosearch/testset_full_videos.json",
        help="Path to the JSON file containing the list of tasks."
    )
    parser.add_argument(
        '--bootstrap_dir',
        type=str,
        default="none",
        help="Path to non-video bootstrap directory"
    )
    parser.add_argument(
        '--scorefile',
        type=str,
        default="none",
        help="Path to video-relevancy score file"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default="mean",
        help="The mode of evaluation, mean or max"
    )
    parser.add_argument(
        '--max_k',
        type=int,
        default=3,
        help="Maximum k to calculate for pass@k."
    )
    parser.add_argument(
        '--score_th',
        type=int,
        default=8,
        help="Score threshold"
    )

    args = parser.parse_args()

    tasks_by_cat_to_video = get_task_list(args.task_file)
    if not tasks_by_cat_to_video:
        return

    task_video_to_score = None
    if args.scorefile != "none" and os.path.exists(args.scorefile):
        with open(args.scorefile) as fin:
            task_video_to_score = json.load(fin)

    stats, unfinished, per_task_stats = calculate_pass(
        args.results_dir,
        tasks_by_cat_to_video,
        mode=args.mode,
        task_video_to_score=task_video_to_score,
        bootstrap_dir=args.bootstrap_dir,
        score_th=args.score_th,
        max_k=args.max_k,
    )
    print("\n===== Category-level results =====")
    print_results_table(stats, args.max_k)

    grouped_stats = aggregate_stats_by_group(stats)
    print("\n===== Group-level results =====")
    print_results_table(grouped_stats, args.max_k)
    with open(args.task_file.replace(".json", "_unfinished.json"), "w") as f:
        json.dump(unfinished, f, indent=4)
    with open(os.path.join(args.results_dir, "stats.json"), "w") as fout:
        json.dump(per_task_stats, fout, indent=4)

if __name__ == "__main__":
    main()