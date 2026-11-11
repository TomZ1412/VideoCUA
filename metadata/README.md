# Task-to-video metadata

Each JSON maps a base task ID to a list of tutorial YouTube video IDs. IDs are
stored without file paths or .mp4 suffixes. Task/video variant numbers are not
part of the base task ID. A video may be associated with more than one task.

| Mapping | Tasks | Task–video associations |
|---|---:|---:|
| cuagym_task_to_video.json | 220 | 2,470 |
| opencua_task_to_video.json | 5,584 | 22,844 |
| osworld_task_to_video.json | 310 | 1,940 |

OSWorld is generated from its experiment mapping. The CUA-Gym task set is
selected directly from cuagym_220_cherrypick.json, and video IDs are resolved
from cuagym_new_task_to_video.json. The OpenCUA mapping is retained from the previous release.
Only duplicate IDs within each task are removed. This metadata does not contain
videos, descriptions, training annotations or model outputs. Resolve IDs to your
locally downloaded media when preparing loader inputs.
