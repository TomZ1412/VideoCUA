# Guidance extraction

Use the existing Qwen3.5 `test.sh` in
`../../vsvex_train/qwen35_cua_multirl_dualora/`, with the guide adapter
`CHECKPOINT/adapter_2`. See that directory's scripts/README_test.md.

`get_video_desc.py` is copied from the original experiment directory. Configure
MAPPING_PATH, place combined decoding predictions in test_results.json in the
working directory, then run the script. It outputs vs2_desc_fromSFT.json.
Matching uses task ID and video filename, allowing different media roots.
The original script also writes resolved video paths back to MAPPING_PATH.
