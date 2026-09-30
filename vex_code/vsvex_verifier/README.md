# video-SALMONN Verifier

This directory contains the video-SALMONN verifier used to score narrated execution videos. It
loads the verifier dataset produced by `../vsvex_inference/verification/gather_data.py`, evaluates
candidate scores from 0 to 9, and writes their expected score when `--expectation` is enabled.

Run from the repository root:

```bash
bash vex_code/vsvex_verifier/run_verifier.sh \
    --run_name vex_verifier \
    --dataset /path/to/verifier_input.json \
    --model /path/to/verifier_model \
    --model_base /path/to/video_salmonn_2_plus \
    --lora_ckpt /path/to/verifier_lora \
    --max_frames 768 \
    --max_pixels 128000 \
    --total_timestamps 256 \
    --model_type dense \
    --expectation
```

Set `GPU_COUNT` or pass `--gpu_count` to control local data parallelism. Per-rank outputs and the
merged `test_results.json` are written to `output/test/<run_name>/` by default. Set `OUTPUT_DIR` to
change the output root.

The verifier model and LoRA checkpoint are not included in the repository.
