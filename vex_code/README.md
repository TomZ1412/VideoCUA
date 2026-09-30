# VEX Code

This directory contains the training, agent evaluation, trajectory verification, and result
selection code for video-SALMONN-VEX. For installation, data and checkpoint preparation, AGGO
training, and benchmark launch commands, see the [project README](../README.md).

## Code map

| Component | Location | Purpose |
| --- | --- | --- |
| AGGO training | `vsvex_train/cua_multirl_dualora/` | Dual-LoRA training with action-likelihood rewards |
| Public launchers | `scripts/` | AGGO training and OSWorld/CUA-Gym evaluation entry points |
| video-SALMONN-VEX agent | `vsvex_inference/models/`, `vsvex_inference/tests/test_vss_local*.py` | Model loading and benchmark execution |
| Gemini agent | `vsvex_inference/mm_agents/gemini/`, `vsvex_inference/tests/test_gemini_local*.py` | Gemini-based benchmark execution |
| Verification | `vsvex_inference/verification/` | Convert trajectories to narrated videos and score them |
| video-SALMONN verifier | `vsvex_verifier/` | Score narrated trajectories from 0 to 9 |
| Result selection | `scorer/` | Aggregate repeated trials and apply VEX selection policies |

## Verification pipeline

Run the commands in this section from `vsvex_inference/`.

### 1. Collect trajectories

Convert evaluation result folders into a trajectory manifest:

```bash
python verification/get_gpt_data.py \
    --whitelist /path/to/task_manifest.json \
    --basepath /path/to/results/RUN_ID \
    --examples-root /path/to/OSWorld/evaluation_examples/examples \
    --output /path/to/workspace/trajectories.json \
    --model vs2 \
    --nsample 1
```

Increase `--nsample` when multiple executions are available for each task.

### 2. Build narrated videos

1. Configure `traj_file` and `traj_output` in `verification/get_audio.py`, create the target audio
   directory, and run the script to synthesize each action description with Kokoro TTS.
2. Configure the input manifest and output video paths in `verification/make_video.py`, then run it
   to combine screenshots and audio with FFmpeg.

These two scripts currently use source-level path constants. They require Kokoro TTS, `ffmpeg`, and
`ffprobe`.

### 3. Build verifier input

`verification/gather_data.py` converts the video manifest into the JSON format consumed by the
video-SALMONN verifier. Each item contains the trajectory video, task instruction, task ID,
model/source metadata, and verification prompt. Its input files and video root are currently
configured as source-level constants and must be changed for a new run.

### 4. Score trajectories

For the video-SALMONN verifier, run:

```bash
bash ../vsvex_verifier/run_verifier.sh \
    --interval 1 \
    --run_name "${VERIFIER_RUN_NAME}" \
    --dataset "${VERIFIER_DATASET}" \
    --max_frames 768 \
    --model "${VERIFIER_MODEL}" \
    --model_base "${MODEL_BASE_PATH}" \
    --model_type dense \
    --total_timestamps 256 \
    --max_pixels 128000 \
    --lora_ckpt "${VSS_SFT_CHECKPOINT}" \
    --expectation
```

See [`vsvex_verifier/README.md`](vsvex_verifier/README.md) for launcher details. The verifier code is
included, while its model and LoRA checkpoints have not been released yet.

Alternatively, configure the endpoint, model, input JSON, and output JSON placeholders in
`verification/gemini_verifier.py`, then run:

```bash
python verification/gemini_verifier.py
```

Provide credentials outside version control. The output adds a `verification` score to each item.

### 5. Select the final trajectory

The released scripts in `scorer/` support aggregation across repeated samples and videos.
`scorer/filter_tasks_video_ordered_grouped.py` implements the current grouped ordered policy:

- begin with a no-video execution;
- stop when its score reaches `threshold`;
- otherwise retry with video guidance until `max_trial` is reached; and
- if no trial reaches the threshold, retain the highest-scoring execution.

The scorer currently uses source-level paths and policy constants. Update them before running on a
new result directory. The legacy standalone `filter_tasks_video.py` and
`filter_tasks_video_ordered.py` scripts referenced in the original workflow are not included; use
the released aggregation scripts and grouped ordered implementation instead.
