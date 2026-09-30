import json


# verifier_result = "output/test/LLM_verifier_vs2/test_results.json"
# verifier_result = "output/test/LLM_verifier_vs2_5samples_expectation/test_results.json"
# verifier_result = "output/test/LLM_verifier_vs2_desc_expectation/test_results.json"
# verifier_result = "video_eval_data_vs2_desc_gemini_verify.json"
# verifier_result = "video_eval_data_5sample_gemini_verify.json"
# verifier_result = "code/OSWorld-self-evolve/tools/gemini.json"
# verifier_result = "code/OSWorld-self-evolve/tools/vs2_agentnet_dualLoRA_plan.json"

# verifier_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_5samples_dualboth/test_results.json"

# # verifier_result = "code/thu_qwenvl3/output/test/LLM_verifier_gemini_35_realvideos_desc_dualLoRA_both/test_results.json"

# backup_result = "output/test/LLM_verifier_vs2/test_results.json"

# padding_result = "output/test/LLM_verifier_vs2_5samples_expectation/test_results.json"

# # padding_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_5samples_dualplan/test_results.json"

# testfile = "../testset_full.json"

# description_file = "output/test/vs2_geminiSFT_key_to_traj_osworld_realvideos/vs2_desc_fromSFT.json"

# verifier_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_5samples_dualboth/test_results.json"
# verifier_result = "code/OSWorld-self-evolve/data/workspace/video_eval_data_vs2_5samples_dualboth_gemini_verify.json"
# verifier_result = "code/OSWorld-self-evolve/tools/vs2_agentnet_dualLoRA_both_qwen3vl_verify.json"

# backup_result = "video_eval_data_gemini_verify.json"
# backup_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_5samples_dualboth/test_results.json"
# padding_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_5samples_dualboth/test_results.json"
# backup_result = "output/test/LLM_verifier_vs2/test_results.json"
# padding_result = "output/test/LLM_verifier_vs2_5samples_expectation/test_results.json"
# padding_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_5samples_dualplan/test_results.json"
# testfile = "../testset_full.json"
# description_file = "output/test/vs2_agentnet_fromagentnet_GRPO_traindesc_realvideo_multiRL_bothLoRA_PPL_sample9_3_desc_g3_epoch2/vs2_desc_fromSFT.json"
testfile = "OSWorld-self-evolve/cuagym_data/cuagym_data_350.json"
description_file = "output/test/vs2_agentnet_fromagentnet_GRPO_traindesc_realvideo_multiRL_bothLoRA_PPL_sample9_3_desc_cuagym/vs2_desc_fromSFT.json"
# description_file = "output/test/vs2_geminiSFT_key_to_traj_osworld_realvideos/vs2_desc_fromSFT.json"
# description_file = "output/test/vs2_agentnet_fromagentnet_GRPO_traindesc_realvideo_multiRL_bothLoRA_PPL_sample9_3_desc_cuagym/vs2_desc_fromSFT.json"

# backup_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_cuagym220_fixed/test_results.json"
# padding_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_agentnet_trainproj_cuagym220_fixed/test_results.json"
# verifier_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_agentnet_cuagym220_dualboth_fixed/test_results.json"
# backup_result = "code/OSWorld-self-evolve/data/workspace/video_eval_data_gemini35_cuagym220_fixed_gemini_verify.json"
# padding_result = "code/OSWorld-self-evolve/data/workspace/video_eval_data_gemini35_cuagym220_3samples_fixed_gemini_verify.json"
# verifier_result = "code/OSWorld-self-evolve/data/workspace/video_eval_data_gemini35_cuagym220_dualboth_fixed_gemini_verify.json"

# backup_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_agentnet_trainproj_cuagym220_fixed/test_results.json"
# padding_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_agentnet_trainproj_cuagym220_fixed/test_results.json"
# verifier_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_agentnet_trainproj_cuagym220_fixed/test_results.json"

# backup_result = "output/test/LLM_verifier_vs2/test_results.json"
# padding_result = "output/test/LLM_verifier_vs2_5samples_expectation/test_results.json"
# verifier_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_agentnet_dualboth_g3e2/test_results.json"

backup_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_cuagym220_fixed/test_results.json"
padding_result = "code/thu_qwenvl3/output/test/LLM_verifier_vs2_agentnet_trainproj_cuagym220_fixed/test_results.json"
verifier_result = "code/OSWorld-self-evolve/data/workspace/video_eval_data_vs2_5samples_cuagym_dualboth_g3e2_gemini_verify.json"

nsample = 1
threshold = 7
max_trial = 3


# ==========================================================
# Load description lengths
# ==========================================================

with open(description_file) as fin:
    data = json.load(fin)

task_to_lengths = {}

for task, desc in data.items():
    task_to_lengths[task] = len(desc.split())

    if "I can not get useful knowledge" in desc:
        task_to_lengths[task] = 1


# ==========================================================
# Load finished map
# ==========================================================

finished_map = {}

with open(
    "code/"
    "OSWorld-self-evolve/data/workspace/"
    "traj_data_vs2_5samples_dualboth.json"
) as fin:
    trajdata = json.load(fin)

for datapiece in trajdata:

    action = "".join(
        datapiece["steps"][-1]["action"].split("<tool_call>")[1:]
    )

    if "terminate" not in action:
        finished_map[datapiece["task"]] = False
    else:
        finished_map[datapiece["task"]] = True


# ==========================================================
# Load original tasks
# ==========================================================

with open(testfile) as fin:
    origtasks = json.load(fin)


# ==========================================================
# Load backup results
# ==========================================================

with open(backup_result) as fin:
    backup_scores = json.load(fin)

done_tasks = {}
unfinished_tasks = {}

for datapiece in backup_scores:

    if "qwen35" not in datapiece["video"]:

        task_id = datapiece["task_id"]

        if task_id not in done_tasks:
            done_tasks[task_id] = []

        prediction = (
            float(datapiece["verification"]["score"])
            if "gemini" in backup_result
            else float(datapiece["pred"])
        )

        reference = (
            float(datapiece["conversations"][-1]["value"])
            if "gemini" in backup_result
            else float(datapiece["ref"])
        )

        done_tasks[task_id].append(
            [prediction, reference, 100000]
        )


# ==========================================================
# Load verifier results
# ==========================================================

with open(verifier_result) as fin:
    scores = json.load(fin)

for datapiece in scores:

    desc_len = (
        task_to_lengths[datapiece["task_id"]]
        if datapiece["task_id"] in task_to_lengths
        else 1
    )

    task_id = (
        datapiece["task_id"]
        .split("_video_")[0]
        .split("_sample_")[0]
    )

    if task_id not in done_tasks:
        done_tasks[task_id] = []

    prediction = (
        float(datapiece["verification"]["score"])
        if "gemini" in verifier_result
        else float(datapiece["verification"]["score_float"])
        if "qwen3" in verifier_result
        else float(datapiece["pred"])
    )
    if "verification" in datapiece and "result_label" in datapiece["verification"] and  datapiece["verification"]["result_label"] is None:
        continue

    reference = (
        float(datapiece["conversations"][-1]["value"])
        if "gemini" in verifier_result
        else float(datapiece["verification"]["result_label"])
        if "qwen3" in verifier_result
        else float(datapiece["ref"])
    )

    done_tasks[task_id].append(
        [prediction, reference, desc_len]
    )


# ==========================================================
# Padding
# ==========================================================

if padding_result != "":

    with open(padding_result) as fin:
        padding_scores = json.load(fin)

    for datapiece in padding_scores:

        task_id = (
            datapiece["task_id"]
            .split("_video_")[0]
            .split("_sample_")[0]
        )

        prediction = (
            float(datapiece["verification"]["score"])
            # if "gemini" in padding_scores
            if "gemini" in padding_result
            else float(datapiece["pred"])
        )

        reference = (
            float(datapiece["conversations"][-1]["value"])
            # if "gemini" in padding_scores
            if "gemini" in padding_result
            else float(datapiece["ref"])
        )

        if len(done_tasks[task_id]) < max_trial:
            done_tasks[task_id].append(
                [prediction, reference, 0]
            )


# ==========================================================
# Category mapping
# ==========================================================

# CATEGORY_GROUPS = {
#     "os": "OS",

#     # Support both naming conventions
#     "calc": "Office",
#     "impress": "Office",
#     "writer": "Office",

#     "libreoffice_calc": "Office",
#     "libreoffice_impress_verified": "Office",
#     "libreoffice_writer": "Office",

#     "chrome": "Daily",
#     "vlc": "Daily",
#     "thunderbird": "Daily",

#     "vscode": "Professional",
#     "vs-code": "Professional",
#     "vs_code": "Professional",
#     "gimp": "Professional",

#     "multi_apps": "Workflow",
# }

CATEGORY_GROUPS = {
    "os": "OS",

    # Support both naming conventions

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
        "PDF",
    ]
}


# ==========================================================
# Main statistics
#
# IMPORTANT:
# This block intentionally follows the FIRST script's
# original logic exactly.
# ==========================================================

total_score = 0
total_tasks = 0
count = 0


for cat, tasks in origtasks.items():

    group = CATEGORY_GROUPS.get(cat)

    if group is None:
        print(f"[Warning] Unknown category: {cat}")

    for task in tasks:
        if task not in done_tasks:
            continue

        # Same as first code
        all_tasks = sorted(
            done_tasks[task],
            key=lambda tup: tup[2],
            reverse=True
        )

        cursor = 0
        last_diff = 0

        current_score = all_tasks[0][0]

        # Same as first code
        while (
            current_score < threshold
            and cursor < max_trial
            and cursor < len(done_tasks[task]) - 1
        ):
            cursor += 1

            prev_score = current_score

            current_score = all_tasks[cursor][0]


        # ==================================================
        # EXACT same decision logic as first code
        # ==================================================

        if cursor == max_trial:

            score_ranking = sorted(
                done_tasks[task][:max_trial],
                key=lambda tup: tup[0]
            )

            final_score = score_ranking[-1][1]
            num_executions = max_trial

        elif cursor == len(done_tasks[task]) - 1:

            score_ranking = sorted(
                done_tasks[task],
                key=lambda tup: tup[0]
            )

            final_score = score_ranking[-1][1]
            num_executions = len(done_tasks[task])

        else:

            final_score = all_tasks[cursor][1]
            num_executions = cursor + 1


        # ==================================================
        # Overall statistics
        # ==================================================

        total_score += final_score
        count += num_executions
        total_tasks += 1


        # ==================================================
        # Group statistics
        #
        # Only add grouping.
        # Does NOT affect overall statistics.
        # ==================================================

        if group is not None:

            group_stats[group]["total_score"] += final_score

            group_stats[group]["total_tasks"] += 1

            group_stats[group][
                "total_executions"
            ] += num_executions


# ==========================================================
# Overall results
# ==========================================================

print("=" * 60)
print("Overall")
print("=" * 60)

print(f"total tasks: {total_tasks}")

print(
    f"total executions per task: "
    f"{count / total_tasks}"
)

print(
    f"average score: "
    f"{total_score / total_tasks}"
)


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
        stats["total_score"]
        / n_tasks
    )

    avg_exec = (
        stats["total_executions"]
        / n_tasks
    )

    print(
        f"{group:15s} | "
        f"tasks: {n_tasks:4d} | "
        f"exec/task: {avg_exec:.4f} | "
        f"avg score: {avg_score:.4f}"
    )

