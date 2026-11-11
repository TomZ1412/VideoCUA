# Qwen3.5 base verifier

Copied from the existing `thu_qwenvl35` implementation: test.sh, verify.py,
runtime.py and their required local dependencies. No new verifier implementation
or Shell wrapper is introduced. The previous Qwen3-VL snapshot has been removed.

Use Qwen3.5 base weights directly. **Do not pass --lora_ckpt** for base-model
verification. By default the score is the probability-weighted expectation over
score digits 0–9. Inputs can be video conversations or screenshot/text trajectories.

## Native Qwen3.5 base

```bash
cd vex_code/vsvex_verifier
bash test.sh --model /path/to/Qwen3.5-9B \
  --dataset /absolute/path/to/video_eval_data.json \
  --run_name qwen35_base_verifier --num_gpus 1 \
  --expectation true --no_audio --interval 1 --max_frames 768 --max_pixels 128000
```

Native Qwen3.5 base has no audio encoder; --no_audio disables audio extraction
from videos while preserving the visual frames. For screenshot/text trajectory
inputs, --no_audio is unnecessary. To verify with audio, supply a full Qwen3.5
**audio-base** checkpoint (and --model_base for its processor if needed), omit
--no_audio, and still omit --lora_ckpt to use base weights without adapters.

The launcher uses independent model replicas per GPU and produces
`output/test/<run_name>/test_results.json`, compatible with the existing scorer.
Use a new run_name/output path for each run. Set --num_gpus explicitly for local
runs; the copied launcher also supports ARNOLD multi-node configuration.

## Dependencies and check

Python 3.11+, a compatible CUDA/PyTorch stack, Qwen3.5-capable Transformers,
TorchCodec and FFmpeg. See requirements.txt for the source environment dependency
constraints. Do not import this directory's qwenvl package together with another
model snapshot in one process.

For an input-format check without loading model weights:

```bash
python verify.py --model unused --dataset /absolute/path/to/video_eval_data.json \
  --output /tmp/qwen35_verifier_input_check.json --dry-run --limit 2 --no_audio
```

The copied implementation supports optional adapters for other experiments,
but the base-verifier configuration above does not load one. This release-copy
check covers syntax and input formatting, not GPU inference or score quality.

## Anonymous experiment launcher

`run_verifier.sh` calls the existing test.sh with the supplied experiment settings:
8 GPUs, interval 1, maximum 768 frames, dense model, max_pixels 128000, and
probability expectation. It contains no private paths or checkpoint identifiers.

```bash
export MODEL_PATH=YOUR_QWEN35_MODEL_CHECKPOINT_PATH
export MODEL_BASE=YOUR_QWEN35_PROCESSOR_PATH
export VERIFIER_DATA=YOUR_VERIFIER_INPUT_JSON_PATH
# Optional: set this to reproduce the supplied caption-adapter configuration.
export LORA_CKPT=YOUR_CAPTION_ADAPTER_PATH
RUN_NAME=qwen35_verifier bash run_verifier.sh
```

For base-only verification, leave LORA_CKPT unset (the launcher passes No).
For native Qwen3.5 without audio support, set NO_AUDIO=true. Qwen3.5 audio-base
inputs can retain audio. Actual verifier output records are shown in
`../../example/verification.json` (pred/ref/score_probs). This is not verifier
input. For video input, use the conversations/video/use_audio schema produced
by vsvex_inference/verification/gather_data.py. Set use_audio=true only with an
audio-enabled checkpoint.

NUM_GPUS, INTERVAL, RUN_NAME, MAX_FRAMES, MODEL_TYPE, MAX_PIXELS and EXPECTATION
can override the defaults. Additional CLI arguments are passed to test.sh.
The relevant base and optional adapter weights will be published later;
checkpoint download links will be added when available.
