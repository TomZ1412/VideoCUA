# Qwen3.5 SFT and AGGO training

The launchers reproduce the supplied experiment hyperparameters with anonymous,
environment-based data and checkpoint configuration. They call the existing
`qwen35_cua_multirl_dualora/train.sh`; no training implementation is duplicated.

## Training stages and data formats

| Launcher | Purpose | Input format example | Defaults |
|---|---|---|---|
| sft_action.sh | Action SFT, including visual projector training | ../../example/train_sft_action.json | 9 epochs, save every 2000 steps, --train_proj |
| sft_desc.sh | Tutorial guidance SFT | ../../example/train_sft_desc.json | 3 epochs, save every 2000 steps |
| aggo.sh | Joint guide/action dual-LoRA AGGO | ../../example/train_aggo.json | 2 epochs, save every 100 steps, PPL reward, RLOO |

Each example contains two sampled records with media paths replaced by
placeholders. They illustrate the JSON schema and supervised assistant targets;
they are not complete training/validation datasets and do not include media.
Supply separate full training and validation JSONs through TRAIN_DATA and
VALIDATION_DATA. Paths should be absolute because the launchers change directory.

Action SFT uses screenshot/action conversations. Description SFT uses tutorial
videos, initial screenshots and nonempty reference descriptions. AGGO combines
tutorial video, task instruction, initial screenshot and oracle action history.
The guidance_osworld.json examples instead have empty
assistant outputs: those are inference inputs, **not description-SFT targets**.

## Required configuration

```bash
export MODEL_PATH=YOUR_FULL_QWEN35_AUDIO_CHECKPOINT_PATH
export MODEL_BASE=YOUR_QWEN35_PROCESSOR_PATH
export TRAIN_DATA=YOUR_TRAIN_JSON_PATH
export VALIDATION_DATA=YOUR_VALIDATION_JSON_PATH
export LORA_CKPT=YOUR_INITIALIZATION_ADAPTER_PATH
```

MODEL_PATH must be a full, compatible Qwen3.5 audio checkpoint. MODEL_BASE supplies
processor/tokenizer assets. For SFT, LORA_CKPT is the initial caption/audio-video
adapter used by the supplied configuration. For AGGO, LORA_CKPT is the action-SFT
adapter and LORA_CKPT_2 is the description-SFT adapter.

```bash
# Run from this directory, after exporting the configuration for each stage.
RUN_NAME=sft_action bash sft_action.sh
RUN_NAME=sft_desc bash sft_desc.sh

export MODEL_PATH=YOUR_ACTION_SFT_FULL_BASE_PATH
export LORA_CKPT=YOUR_ACTION_SFT_ADAPTER_PATH
export LORA_CKPT_2=YOUR_DESCRIPTION_SFT_ADAPTER_PATH
export TRAIN_DATA=YOUR_AGGO_TRAIN_JSON_PATH
export VALIDATION_DATA=YOUR_AGGO_VALIDATION_JSON_PATH
RUN_NAME=aggo bash aggo.sh
```

The Qwen3.5 branch calls this dual-LoRA algorithm `--train_type cua_grpo`;
`--train_type grpo` selects a different trainer. The AGGO launcher therefore uses
cua_grpo with planner_type=plan, adapter_update=both, reward_func=ppl, and RLOO.
GRPO_NUM_SAMPLE=3 corresponds to nine guide candidates in this implementation.

## Optional environment settings

All three launchers accept RUN_NAME, INTERVAL (1.0), MAX_FRAMES (512),
LEARNING_RATE (2e-5), MODEL_TYPE (dense), LORA_R (128), LORA_ALPHA (256),
MAX_PIXELS_IMG (2076192), MAX_PIXELS (128000), EPOCHS, SAVE_STEPS and REPORT_TO
(none). AGGO additionally accepts GRPO_NUM_SAMPLE (3) and ACTION_SAMPLES (1).
The defaults for EPOCHS / SAVE_STEPS are shown in the table above.

TOTAL_TIMESTAMPS defaults to 256 to retain the supplied launch configuration;
the native Qwen3.5 entry accepts this legacy option but does not apply the limit.
All launchers enable validation and LoRA, disable internal HDFS upload, and pass
additional command-line arguments to the existing train.sh. Hardware/process
configuration remains controlled by that launcher's existing environment options.

## Checkpoint release

The audio-aligned base, caption-initialization adapter, action-SFT checkpoint,
description-SFT checkpoint and final AGGO weights **will be published later**.
Download links will be added when released. No weights are included in this
repository. Supply compatible local checkpoints until those links are available.
