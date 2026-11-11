# Verification

These are existing experiment scripts, not a newly introduced pipeline wrapper.
Configure the input/output paths in the scripts before use.

1. Collect benchmark trajectories using get_gpt_data.py (OSWorld) or
   get_gpt_data_cuagym.py. The former has --whitelist, --basepath,
   --examples-root and --output options.
2. Use get_audio.py for Kokoro narration. Its copied version accepts path
   configuration through VEX_TRAJ_FILE, VEX_TRAJ_OUTPUT and VEX_AUDIO_DIR.
3. Use make_video.py for ffmpeg rendering. Configure VEX_INPUT_JSON,
   VEX_OUTPUT_JSON and VEX_VIDEO_DIR.
4. Configure the existing gather_data.py to convert rendered trajectory records
   to the verifier input format. This retains its original experiment I/O logic.
5. Run ../../vsvex_verifier/test.sh with Qwen3.5 base weights, no LoRA,
   --expectation true, and --no_audio for native base video input. The existing
   test.sh supports --num_gpus 1 for a single GPU. See the verifier README.
6. Use ../../scorer/filter_tasks_video_ordered_grouped_fixed.py to select trials.

The screenshot/text comparison is qwen35_cua_verifier.py; the Gemini video
comparison is gemini_verifier.py. score_verifier.py contains optional PCC/AUROC
analysis. These are copied tools and may require local input/provider settings.

This is offline verification and trial selection over existing rollouts, not
an online environment scheduler. Full data, videos and result files are omitted.

All machine/experiment-specific input and output paths are marked as
`YOUR_XXX_PATH`. Replace these placeholders with your own paths or supply the
existing command-line/environment options before running the scripts.
