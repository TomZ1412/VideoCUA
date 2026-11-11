"""Shared training Dataset preprocessing for offline and HTTP inference."""
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, WhisperFeatureExtractor

from qwenvl.data.data_qwen import LazySupervisedDataset
from qwenvl.train.argument import DataArguments

DEFAULT_PROMPT = "Describe the video in detail, including the audio and its alignment with the visual content."


def add_media_args(parser):
    parser.add_argument('--interval', '--base-interval', type=float, default=0.1)
    parser.add_argument('--min-frame', '--video-min-frames', type=int, default=4)
    parser.add_argument('--max-frame', '--video-max-frames', type=int, default=128)
    parser.add_argument('--min-pixel', '--video-min-frame-pixels', type=int, default=4096)
    parser.add_argument('--max-pixel', '--video-max-frame-pixels', type=int, default=176400)
    parser.add_argument('--no-audio', action='store_true')


def media_options(args):
    if args.interval <= 0 or not 1 <= args.min_frame <= args.max_frame:
        raise ValueError('Require interval > 0 and 1 <= min-frame <= max-frame')
    if not 1024 <= args.min_pixel <= args.max_pixel:
        raise ValueError('Require 1024 <= min-pixel <= max-pixel')
    return dict(base_interval=args.interval, video_min_frames=args.min_frame,
                video_max_frames=args.max_frame, video_min_frame_pixels=args.min_pixel,
                video_max_frame_pixels=args.max_pixel)


def make_dataset(model, samples, options):
    tokenizer = AutoTokenizer.from_pretrained(model)
    processor = AutoProcessor.from_pretrained(model)
    args = DataArguments(**options)
    args.image_processor = processor.image_processor
    args.video_processor = processor.video_processor
    audio_path = Path(model) / 'audio_processor'
    args.audio_processor = (WhisperFeatureExtractor.from_pretrained(audio_path)
                           if audio_path.is_dir() else WhisperFeatureExtractor(feature_size=128))
    return LazySupervisedDataset(tokenizer, args, inference=True, samples=samples)


def load_samples(data=None, video_dir=None, prompt=DEFAULT_PROMPT, no_audio=False):
    if data:
        path = Path(data)
        samples = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                   if path.suffix == '.jsonl' else json.loads(path.read_text()))
        # Resolve relative media paths against the annotation file.
        for row in samples:
            for key in ('video', 'audio', 'image'):
                if isinstance(row.get(key), str) and not Path(row[key]).is_absolute():
                    row[key] = str((path.parent / row[key]).resolve())
    else:
        extensions = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
        paths = sorted(p for p in Path(video_dir).rglob('*') if p.suffix.lower() in extensions)
        samples = [dict(video=str(p.resolve()), use_audio=not no_audio,
                        conversations=[{'from': 'human', 'value': '<video>\n' + prompt}]) for p in paths]
    if not samples:
        raise ValueError('No input examples found')
    for row in samples:
        if no_audio:
            row['use_audio'] = False
    return samples


def pack_item(item):
    """One bundle preserves the order of all visual and audio embeddings."""
    keys = ('pixel_values_videos', 'video_grid_thw', 'audio_feature',
            'pixel_values', 'image_grid_thw')
    bundle = {key: item[key].cpu() for key in keys if item.get(key) is not None}
    bundle['av_token_ids'] = item['input_ids'][0].cpu()
    return bundle


def vllm_prompt(item):
    result = {'prompt_token_ids': item['input_ids'][0].tolist()}
    if any(item.get(k) is not None for k in ('pixel_values_videos', 'pixel_values', 'audio_feature')):
        result['multi_modal_data'] = {'video': pack_item(item)}
    return result


class AudioVisualDataset(torch.utils.data.Dataset):
    """Construct processors/TOS clients inside each spawned DataLoader worker.

    LazySupervisedDataset holds a client with thread locks, so passing a live
    instance through multiprocessing's pickle is unsafe.
    """
    def __init__(self, model, samples, options):
        self.model = model
        self.samples = samples
        self.options = options
        self._dataset = None

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        return {**self.__dict__, '_dataset': None}

    def _get_dataset(self):
        if self._dataset is None:
            self._dataset = make_dataset(self.model, self.samples, self.options)
        return self._dataset

    def __getitem__(self, index):
        return self._get_dataset()[index]

    def __getitems__(self, indices):
        return self._get_dataset().__getitems__(indices)
