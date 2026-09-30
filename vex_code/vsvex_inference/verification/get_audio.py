from kokoro import KPipeline
import soundfile as sf
import torch
import soxr
from tqdm import tqdm
import os
import json


def generate(text, pipeline, outpath):
    generator = pipeline(text, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        audio_16k = soxr.resample(audio, 24000, 16000)
        sf.write(outpath, audio_16k, 16000)

pipeline = KPipeline(lang_code='a')

# traj_file = "traj_data_vs2_5samples_dualplan.json"
# traj_output = "traj_data_audio_s2_5samples_dualplan.json"
# traj_file = "traj_data_vs2_5samples_dualplan_v2.json"
# traj_output = "traj_data_audio_vs2_5samples_dualplan_v2.json"
# traj_file = "traj_data_gemini_35_realvideos_desc_dualLoRA_plan.json"
# traj_output = "traj_data_audio_gemini_35_realvideos_desc_dualLoRA_plan.json"
# traj_file = "traj_data_vs2_5samples_cuagym_dualplan.json"
# traj_output = "traj_data_audio_vs2_5samples_cuagym_dualplan.json"
traj_file = "traj_data_vs2_agentnet_cuagym220_dualboth_fixed.json"
traj_output = "traj_data_audio_vs2_agentnet_cuagym220_dualboth_fixed.json"

with open(traj_file) as fin:
    data = json.load(fin)

for datapiece in tqdm(data):
    for k, step in enumerate(datapiece["steps"]):
        outpath = os.path.join("audio_vs2_agentnet_cuagym220_dualboth_fixed", f"{datapiece['task']}_{k}.wav")
        if not os.path.exists(outpath):
            generate(step["description"], pipeline, outpath)
        step["audio"] = outpath

with open(traj_output, "w") as fout:
    json.dump(data, fout, indent=4)
