# VS2 verifier with a Qwen3.5 backbone

This verifier uses **video-SALMONN 2 (VS2), with Qwen3.5 as its language/vision
backbone**, not the native Qwen3.5 model. It retains the VS2 Whisper encoder,
Q-Former/audio projection and audio-visual input pipeline.

## Model configuration

Use the audio-aligned VS2 Qwen3.5 base together with its compatible caption
adapter, matching the experiment configuration. Alternatively, use a fully
merged VS2 checkpoint and set LORA_CKPT=No. The runtime rejects native Qwen3.5
checkpoints without audio_config so the wrong model cannot be selected silently.

```bash
export MODEL_PATH=YOUR_VS2_QWEN35_AUDIOALIGNED_BASE_PATH
export MODEL_BASE=YOUR_VS2_QWEN35_PROCESSOR_PATH
export LORA_CKPT=YOUR_VS2_QWEN35_CAPTION_ADAPTER_PATH
export VERIFIER_DATA=YOUR_VERIFIER_INPUT_JSON_PATH
bash run_verifier.sh
```

The caption adapter above is the VS2 video/audio understanding adapter, not the
AGGO action or guide adapter. Relevant weights will be published later.

## Inputs and scores

Video inputs use conversations/video/use_audio, as produced by
vsvex_inference/verification/gather_data.py. Set use_audio=true for narrated
execution videos. The default launcher does not disable audio. NO_AUDIO=true is
available only for explicitly requested silent-video ablations; it does not
switch the underlying model to native Qwen3.5.

The verifier calculates probabilities over scores 0–9 and reports their weighted
expectation. The existing test.sh distributes samples across independent GPU
replicas. Defaults: 8 GPUs, 1 second frame interval, 768 maximum frames, dense
model, max_pixels=128000 and expectation=true. NUM_GPUS, INTERVAL, RUN_NAME,
MAX_FRAMES, MODEL_TYPE, MAX_PIXELS and EXPECTATION override those settings.

Actual output examples are in ../../example/verification.json. Those records
come from the screenshot/text comparison run, so they illustrate the result
schema rather than the narrated-video input modality. They are not verifier
inputs. Outputs include pred, ref, score_probs and argmax_score and are compatible
with the scorer. Use a fresh output path/run name per run.

## Files and environment

- runtime.py: load the VS2 model/processor and merge the configured adapter.
- verify.py: preprocess inputs and calculate probability-weighted scores.
- test.sh / scripts/test_parallel.py: launch and merge per-GPU verification.

These files originate from the existing thu_qwenvl35 implementation. The release
copy restricts loading to VS2 audio-visual models and makes the required weight
configuration explicit. Use Python 3.11+, compatible Transformers/PyTorch,
TorchCodec and FFmpeg; see requirements.txt. Run in a separate process from other
qwenvl snapshots. Syntax and input-format checks do not validate GPU inference.
