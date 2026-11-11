"""CUA plan/action examples on the Qwen3.5 multimodal preprocessing path."""

import copy
import json
import math
import random

from torch.utils.data import Dataset, Sampler

from qwenvl.data.data_qwen import LazySupervisedDataset, multimodal_token_type_ids


PLAN_TEMPLATE = """<video>
You are given the above video tutorial teaching you how to perform a computer use task. The task is described as follows:
{instruction}

The starting point of the task is shown in the following screenshot:
<image>

Use ONLY the knowledge from the video, describe how to perform the task step by step starting from the current screenshot.
The video may not contain the entire task. It may only contain a part of the task. Just describe the part of the task that is shown in the video.

For example, for a long task where part of it requires you to login with username and password that is included in the video, you should output the following:
When you want to log in, you should first click the login button on the left menu bar. Then, this will direct you to the login page. You need to enter your username and password. You should first click on the textbox with a word "Username" and then enter your username. Then, click on the textbox with a word "Password" and enter your password.

### Rules
1. Do not use your own knowledge. Just describe what you can learn from the video.
2. Always output step by step, and at the end of each step, describe what screen status you are expected to get.
3. Try to gather as much information/procedural knowledge as possible from the video.
4. If the video contains the entire task, describe the entire task.
5. Do not output irrelevant details of the video.

Directly output your step-by-step description.
"""


def _read_samples(paths):
    samples = []
    for path in paths.split(","):
        if not path:
            continue
        if path.endswith(".jsonl"):
            with open(path) as handle:
                samples.extend(json.loads(line) for line in handle if line.strip())
        else:
            with open(path) as handle:
                loaded = json.load(handle)
                samples.extend(loaded if isinstance(loaded, list) else [loaded])
    return samples


def _normalise_video(sample):
    if isinstance(sample.get("video"), list):
        sample["videos"] = [{"video": path} for path in sample.pop("video")]
    if isinstance(sample.get("tos_key"), list):
        sample["videos"] = [{"tos_key": path} for path in sample.pop("tos_key")]


def _paired_sources(source):
    images = source.get("images", source.get("image"))
    if isinstance(images, str):
        images = [images]
    if not images or not isinstance(images, list):
        raise ValueError("CUA samples require an images list containing the current screenshot")
    if not any(key in source for key in ("video", "videos", "tos_key")):
        raise ValueError("CUA samples require a tutorial video")
    turns = source.get("conversations", [])
    if len(turns) < 2 or turns[-1].get("from") not in ("gpt", "assistant"):
        raise ValueError("CUA samples require a final assistant action")

    plan = copy.deepcopy(source)
    _normalise_video(plan)
    plan.pop("images", None)
    plan["image"] = images[0]
    plan["conversations"] = [
        {"from": "human", "value": PLAN_TEMPLATE.format(
            instruction=source["instruction"])},
        {"from": "gpt", "value": "placeholder"},
    ]
    plan["use_audio"] = bool(source.get("use_audio", False))

    action = copy.deepcopy(source)
    action.pop("images", None)
    action["image"] = images
    for key in ("video", "videos", "tos_key", "use_audio", "audio", "tos_audio"):
        action.pop(key, None)
    action["conversations"][0]["value"] = action["conversations"][0]["value"].replace("<video>\n", "")
    return plan, action, turns[-1]["value"]


class CUADualLoRADataset(Dataset):
    def __init__(self, tokenizer, data_args, paths):
        raw = _read_samples(paths)
        paired = [_paired_sources(sample) for sample in raw]
        plan_args = copy.copy(data_args)
        action_args = copy.copy(data_args)
        plan_args.thinking = False
        action_args.thinking = False
        self.plan = LazySupervisedDataset(
            tokenizer, plan_args, inference=True, samples=[pair[0] for pair in paired])
        self.action = LazySupervisedDataset(
            tokenizer, action_args, samples=[pair[1] for pair in paired])
        self.references = [pair[2] for pair in paired]
        self.audio_modalities = [bool(pair[0].get("use_audio", False)) for pair in paired]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        plan = self.plan[index]
        # The generic SFT dataset retries a failed decode with a random sample.
        # That would attach the wrong action reference to this plan.
        action = self.action._get_item([self.action.list_data_dict[index]], no_meta=True)
        action["mm_token_type_ids"] = multimodal_token_type_ids(action["input_ids"], self.tokenizer)
        return {"plan": plan, "action": action, "reference": self.references[index]}


def collate_cua(batch):
    if len(batch) != 1:
        raise ValueError("CUA GRPO requires per-device batch size 1")
    return batch[0]


class CUAModalitySampler(Sampler):
    """Global ordering; Accelerate shards each homogeneous world-size group."""

    def __init__(self, audio_modalities, world_size, seed):
        self.groups = {
            modality: [index for index, value in enumerate(audio_modalities) if value == modality]
            for modality in (False, True)
        }
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        if not audio_modalities:
            raise ValueError("CUA dataset is empty")

    def __len__(self):
        return sum(math.ceil(len(indices) / self.world_size) * self.world_size
                   for indices in self.groups.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.groups.values():
            if not indices:
                continue
            shuffled = list(indices)
            rng.shuffle(shuffled)
            shuffled += [shuffled[i % len(shuffled)]
                         for i in range(math.ceil(len(shuffled) / self.world_size)
                                        * self.world_size - len(shuffled))]
            batches.extend(shuffled[i:i + self.world_size]
                           for i in range(0, len(shuffled), self.world_size))
        rng.shuffle(batches)
        for batch in batches:
            yield from batch
