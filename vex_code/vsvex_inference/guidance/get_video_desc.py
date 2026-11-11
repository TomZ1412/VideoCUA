import json
import os
from tqdm import tqdm


MAPPING_PATH = '/path/to/cuagym_new_task_to_video.json'


def normalize_video_path(video_path):
    return os.path.basename(video_path)


with open("test_results.json") as fin:
    data = json.load(fin)

task_video_to_taskname = {}
compat_video_to_taskname = {}

with open(MAPPING_PATH) as fin:
    mapping = json.load(fin)

for taskname, video in mapping.items():
    task = taskname.split("_video_")[0].replace("_verified", "")
    if task not in task_video_to_taskname:
        task_video_to_taskname[task] = {}
        compat_video_to_taskname[task] = {}
    task_video_to_taskname[task][video] = taskname
    compat_video_to_taskname[task][normalize_video_path(video)] = (taskname, video)

taskname_to_desc = {}
updated_mapping = False
for datapiece in tqdm(data):
    task = datapiece["images"][0].split("/")[-2]
    video = datapiece["video"]
    task_videos = task_video_to_taskname[task]
    taskname = task_videos.get(video)

    if taskname is None:
        compat_match = compat_video_to_taskname[task].get(normalize_video_path(video))
        if compat_match is None:
            raise KeyError(f"Cannot match task={task}, video={video}")
        taskname, old_video = compat_match
        if old_video != video:
            mapping[taskname] = video
            task_videos.pop(old_video, None)
            task_videos[video] = taskname
            compat_video_to_taskname[task][normalize_video_path(video)] = (taskname, video)
            updated_mapping = True

    taskname_to_desc[taskname] = datapiece["pred"]

with open("vs2_desc_fromSFT.json", "w") as fout:
    json.dump(taskname_to_desc, fout, indent=4, ensure_ascii=False)

if updated_mapping:
    with open(MAPPING_PATH, "w") as fout:
        json.dump(mapping, fout, indent=4, ensure_ascii=False)
    print(f"Updated mapping file: {MAPPING_PATH}")
else:
    print("No mapping updates needed.")