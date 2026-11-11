# Running the Qwen3.5 VEX code

## Training

See [vsvex_train/README.md](vsvex_train/README.md). The three stage launchers use
MODEL_PATH, MODEL_BASE, TRAIN_DATA, VALIDATION_DATA and LORA_CKPT. AGGO additionally
uses LORA_CKPT_2. Data examples are under ../example/. Weights will be published later.
The existing Qwen3.5 trainer selects dual-LoRA AGGO with train_type=cua_grpo.

## Guidance

The existing vsvex_train/qwen35_cua_multirl_dualora/test.sh decodes guidance with
CHECKPOINT/adapter_2. Use its scripts/README_test.md and the input schema in
../example/guidance_osworld.json. Convert combined
predictions with vsvex_inference/guidance/get_video_desc.py after configuring its
mapping and input/output paths. For silent-video guidance set use_audio=false.

## Benchmarks

Install compatible OSWorld/CUA-Gym environments and task bundles separately.
Place/link the OSWorld checkout at vsvex_inference/OSWorld/ as expected by the
existing runners. Run the following from vsvex_inference/:

```bash
bash ../scripts/eval_qwen35_harness_local.sh
# or
bash ../scripts/eval_qwen35_harness_local_cuagym.sh
```

Required environment variables: RUN_ID, TASK_CONFIG_DIR, MODEL_PATH,
VEX_PROCESSOR_BASE, VOLCENGINE_SUBNET_ID, VOLCENGINE_ZONE_ID, VOLCENGINE_IMAGE_ID,
VOLCENGINE_ACCESS_KEY_ID and VOLCENGINE_SECRET_ACCESS_KEY. Configure the remaining
provider-required variables and VM password/proxy settings for your environment.
LORA_CKPT defaults to No; GUIDE_TRAJ and ORACLE_TRAJ default to none. Use compatible
action adapter_1 and extracted guides for tutorial-conditioned evaluation.
For text guidance without raw tutorial context, set USE_VIDEO=false in OSWorld.

Optional settings: NUM_GPUS=8, TASKS_PER_GPU=3, MAX_STEPS=50, N_HISTORY=5,
NSAMPLES=1, MAX_TOKENS=4096, RESULT_DIR=results. Change these to your compute budget.
The launchers contain no personal paths or cloud resource identifiers.

## Verification and selection

See [vsvex_inference/verification/README.md](vsvex_inference/verification/README.md)
for collecting trajectories, Kokoro narration and ffmpeg rendering. Configure
YOUR_XXX_PATH placeholders or the provided environment/CLI options.

[vsvex_verifier/README.md](vsvex_verifier/README.md) documents run_verifier.sh:
MODEL_PATH, MODEL_BASE and VERIFIER_DATA are required, LORA_CKPT is optional.
Set NO_AUDIO=true when using native Qwen3.5 base on video inputs.

Configure scorer/filter_tasks_video_ordered_grouped_fixed.py with
VEX_VERIFIER_RESULT, VEX_BACKUP_RESULT, VEX_PADDING_RESULT, VEX_TESTFILE and
VEX_DESCRIPTION_FILE; VEX_THRESHOLD=7 and VEX_MAX_TRIAL=3 are the default policy.
It orders and selects existing rollouts; it does not launch new environment trials.

The model packages should run in separate processes/environments so their qwenvl
module names do not conflict. This release does not automatically download weights
or provision benchmark environments.
