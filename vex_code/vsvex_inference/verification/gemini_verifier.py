import time
import copy
import openai
import os
import base64
import json
import concurrent.futures
import threading
import subprocess
import tempfile
from tqdm import tqdm
import regex


pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

prompt = """You are an expert GUI agent reviewer. You will watch the following video of an agent performing a task while describing its action every step.
"""

system_prompt = """
The task is:
{task}

You need to evaluate the completion of this task, with the following score schemes:
0-1 | No meaningful progress; wrong app, crash, infinite repetition, or immediate failure.
2-3 | Correct environment but core action not executed; loops or early abandonment.
4-5 | Main action attempted; deliverable exists but is flawed or missing key constraints.
6-7 | Task complete with minor errors (e.g., slight formatting, naming) or inefficient path.
8-9 | Fully correct final state, all constraints satisfied, clean and robust execution.

You should focus on the following aspects in your evaluation:
- Final frames: Is the exact deliverable visible and correct (saved file, sent email, correct UI state)?
- Constraints: Were ALL specific requirements from the instruction fulfilled?
- Coherence: Penalize unresolved dialogs, unsaved files, and infinite loops.

Carefully analyze the video, then directly provide your final scores in the following format:
{{
    "score": "your predicted score"
}}
"""


def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")

def create_video_clip(original_path):
    if not os.path.exists(original_path):
        return None

    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", original_path,
        # "-t", str(end_time_sec),
        "-vf", "scale=-2:480", 
        "-r", "1",
        "-c:v", "libx264", "-preset", "ultrafast", 
        "-c:a", "aac",
        "-loglevel", "error", 
        temp_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_path
    except subprocess.CalledProcessError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def encode_video_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

def gpt4_caption(videopath, task):
    # params:
    # texts:
    # type: list
    # content: each input text as a element
    client = openai.AzureOpenAI(
        azure_endpoint="https://example.com/v1",
        api_version="2024-03-01-preview",
        api_key=__import__("os").environ.get("OPENAI_API_KEY", ""),
    )
    tmp_path = create_video_clip(videopath)
    base64_video = encode_video_base64(tmp_path)
    data_url = f"data:video/mp4;base64,{base64_video}"
    msg=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": system_prompt.format(task=task)},
            ]
        }
    ]
    completion = client.chat.completions.create(
        model="gemini-3.5-flash", # "gpt-4-turbo-2024-04-09", # "gpt-4o-2024-05-13", # 
        messages=msg,
        seed=2025
    )
    content = completion.choices[0].message.content
    json_obj = pattern.findall(content)
    pred_answer = json.loads(json_obj[0])
    return pred_answer


json_file = 'YOUR_VERIFIER_INPUT_PATH'
output_json = __import__("os").environ.get("VEX_OUTPUT_JSON", 'YOUR_GEMINI_VERIFIER_OUTPUT_PATH')

covered = set()
covered_data = []
if os.path.exists(output_json):
    with open(output_json) as fin:
        covered_data = json.load(fin)
    for datapiece in covered_data:
        covered.add(datapiece["video"])

with open(json_file, 'r') as fp:
    data = json.load(fp)

newdata = []
for datapiece in data:
    if datapiece["video"] not in covered:
        newdata.append(datapiece)
data = newdata + covered_data

def gpt_extract(item):
    try:
        res = gpt4_caption(item["video"], item["instruction"])
        item["verification"] = res
        # item["correctness"] = correctness
        return item
    except Exception as e:
        # raise e
        return item

# data = data[:10]
# gpt_extract(data[0])
# exit()
print(len(data))
# task_video_pairs = task_video_pairs[:5]

# with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
#     responses = list(tqdm(executor.map(gpt_extract, data)))

result = [r for r in data if "verification" in r]
ignore = [r for r in data if "verification" not in r]

print(output_json)
print(len(result), len(ignore))
k = 0

while len(ignore) != 0 and k < 200:
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        responses = list(tqdm(executor.map(gpt_extract, ignore)))
    result += [r for r in responses if "verification" in r]
    ignore = [r for r in responses if "verification" not in r]
    with open(output_json, 'w') as fp:
        json.dump(result, fp, indent=4)
    print(output_json)
    print(len(result), len(ignore))

    k += 1