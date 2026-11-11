"""Lazy, index-based GRPO input: preprocess only prompts assigned to this engine."""
import copy

from torch.utils.data import Dataset

from qwenvl.data.data_qwen import LazySupervisedDataset
from qwenvl.data.video_token_map import VideoTokenMap


class GRPODataset(Dataset):
    def __init__(self, tokenizer, data_args):
        data_args = copy.copy(data_args)
        data_args.thinking = True
        self.media = LazySupervisedDataset(tokenizer, data_args, inference=True)
        for sample, ref in zip(self.media.list_data_dict, self.media.references):
            sample['thinking'] = True
            if ref is None or not str(ref).strip():
                raise ValueError('Every GRPO sample requires a reference answer')
            if len(sample['conversations']) != 1:
                raise ValueError('GRPO requires one user prompt and one reference answer')
        self.modalities = []
        for sample in self.media.list_data_dict:
            visual = 'image' if 'image' in sample else (
                'video' if any(k in sample for k in ('video', 'videos', 'tos_key')) else 'none')
            audio = any(k in sample for k in ('audio', 'tos_audio')) or (
                visual == 'video' and sample.get('use_audio', False))
            self.modalities.append((visual, bool(audio)))
        self.cached_token_lengths = (
            VideoTokenMap.read(data_args.length_cache).lengths_for(self.media.list_data_dict)
            if data_args.length_cache else [1] * len(self.media)
        )

    def __len__(self):
        return len(self.media)

    def __getitem__(self, indices):
        if isinstance(indices, dict):
            return indices
        return {'indices': indices}

    def prepare(self, index):
        return self.media[index]


def collate_prompt_indices(batch):
    if len(batch) != 1:
        raise ValueError('GRPO loader receives one node prompt plan per optimizer step')
    return batch[0]
