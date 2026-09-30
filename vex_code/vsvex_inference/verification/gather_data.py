import json
import os



prompt = """You are an expert GUI agent reviewer. You will watch the following video of an agent performing a task while describing its action every step.
<video>

The task is:
{task}

You need to evaluate the completion of this task, with the following score schemes:
0-1 | No meaningful progress; wrong app, crash, infinite repetition, or immediate failure.
2-3 | Correct environment but core action not executed; loops or early abandonment.
4-5 | Main action attempted; deliverable exists but is flawed or missing key constraints.
6-7 | Task complete with minor errors (e.g., slight formatting, naming) or inefficient path.
8-9 | Fully correct final state, all constraints satisfied, clean and robust execution.

You should focus on the following aspects in your evaluation:
- Final frames: Is the exact deliverable visible and correct (saved file, sent email, correct UI state)?
- Constraints: Were ALL specific requirements from the instruction fulfilled?
- Coherence: Penalize unresolved dialogs, unsaved files, and infinite loops.

Carefully analyze the video, then directly provide your final scores in the following format:
<score> INTEGER_0_TO_9 </score>
"""


# with open("traj_data_video_qwen35.json") as fin:
#     data = json.load(fin)
data = []
workspace_root = os.environ.get("VEX_WORKSPACE_ROOT", "workspace")

with open("traj_data_video_vs2_desc.json") as fin:
    data_vs2 = json.load(fin)

with open("traj_data_qwen35.json") as fin:
    labels = json.load(fin)

task_to_label = {}
for datapiece in labels:
    task_to_label[datapiece["task"]] = datapiece["result"]

video_eval_data = []
for datapiece in data:
    videopath = os.path.join(workspace_root, datapiece["video"])
    if not os.path.exists(videopath):
        import pdb; pdb.set_trace()
    newpiece = {
        "video": videopath,
        "conversations": [
            {
                "from": "human",
                "value": prompt.format(task=datapiece["instruction"])
            },
            {
                "from": "gpt",
                "value": str(task_to_label[datapiece["task"]])
            }
        ],
        "use_audio": True,
        "task_id": datapiece["task"],
        "model": "qwen3.5",
        "instruction": datapiece["instruction"]
    }
    video_eval_data.append(newpiece)

for datapiece in data_vs2:
    videopath = os.path.join(workspace_root, datapiece["video"])
    if not os.path.exists(videopath):
        import pdb; pdb.set_trace()
    newpiece = {
        "video": videopath,
        "conversations": [
            {
                "from": "human",
                "value": prompt.format(task=datapiece["instruction"])
            },
            {
                "from": "gpt",
                "value": str(datapiece["result"])
            }
        ],
        "use_audio": True,
        "task_id": datapiece["task"],
        "model": "vs2",
        "instruction": datapiece["instruction"]
    }
    video_eval_data.append(newpiece)

with open("video_eval_data_vs2_desc.json", "w") as fout:
    json.dump(video_eval_data, fout, indent=4)
