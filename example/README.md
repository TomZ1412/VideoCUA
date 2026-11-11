# Real-source data format examples

Each JSON contains one or two records extracted from actual experiment files. Host
media paths are replaced with YOUR_XXX_PATH placeholders; task content, schemas
and supervised targets are retained. No images, videos, weights or full datasets
are included. Replace paths before running and use full data for experiments.

| Files | Usage |
|---|---|
| train_sft_action.json | Action SFT: screenshot/action conversations |
| train_sft_desc.json | Description SFT: video + screenshot + reference description |
| train_aggo.json | AGGO: task/video pair with oracle action history |
| task_configs_osworld.json | Two actual OSWorld task configurations |
| tasks_osworld.json, tasks_cuagym.json | Category → task/video-variant manifest |
| desc.json | Actual generated task-to-description mapping |
| verification.json | Verifier output records |

The action/description SFT examples come from the corresponding AgentNet training
files; AGGO from the real-video training set. Task, guidance, description and
verification examples come from the OSWorld/CUA-Gym experiment input files.
Verifier output records include reference labels for evaluation/metadata; these
must not be inserted into the verifier prompt as task-completion evidence.
verification.json is sampled from vs2_qwen_harness_traj_aggo_qwen35_verified.json;
it is an output-format example, not an input to the verifier. The source run uses
screenshots and text; its video field is metadata, not proof of video input.

metadata/ contains the full task-to-video-ID mapping; runtime examples use media
paths because that is the format accepted by the existing loaders. Input records
with use_audio=true require a matching audio-enabled checkpoint.
