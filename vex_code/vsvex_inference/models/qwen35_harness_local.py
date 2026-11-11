"""Local Qwen3.5 inference with the unmodified mm_agents.qwen CUA harness."""
import base64
import math
import os
from io import BytesIO

from PIL import Image
import torch

from mm_agents.qwen import QwenAgent


def resize_history_images(images, processor):
    """Keep the newest screenshot unchanged; bound older screenshot pixels."""
    budget = getattr(processor, 'history_max_pixels', 0)
    if not budget or len(images) < 2:
        return images
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size
    minimum = max(ip.min_pixels, factor ** 2)
    if budget < minimum:
        raise ValueError(f'history_max_pixels must be at least {minimum}')
    result = []
    for image in images[:-1]:
        height, width = smart_resize(image.height, image.width, factor=factor,
                                    min_pixels=minimum, max_pixels=budget)
        result.append(image.resize((width, height), Image.Resampling.BICUBIC)
                      if image.size != (width, height) else image)
    return result + images[-1:]


class LocalQwen35Agent(QwenAgent):
    """Keep the QwenAgent harness with local generation and native tutorials.

    predict(), messages, history, screenshots, response handling, and parsing
    are inherited verbatim. In particular, no VS response sanitizer is applied.
    Tutorial videos use native video tokens alongside screenshots/text. Audio
    features are not extracted from the tutorial.
    """

    def __init__(self, *, local_model, processor, enable_thinking=False,
                 video_path=None, max_video_pixels=128000, video_max_frames=512,
                 video_min_frames=2, video_interval=1.0, **kwargs):
        super().__init__(**kwargs)
        self.local_model = local_model
        self.processor = processor
        self.enable_thinking = enable_thinking
        self.max_video_pixels = max_video_pixels
        self.video_max_frames = video_max_frames
        self.video_min_frames = video_min_frames
        self.video_interval = video_interval
        if not 2 <= video_min_frames <= video_max_frames or video_interval <= 0 or max_video_pixels <= 0:
            raise ValueError('Video frames must satisfy 2 <= min <= max; interval and pixels must be positive')
        self._video_cache = {}
        if video_path is not None:
            # QwenAgent keeps examples on the first retained user turn even when
            # history is truncated, so the tutorial stays in context each step.
            self.examples = self._get_examples_video(video_path) + self.examples

    def _get_examples_video(self, example):
        if os.path.isfile(example):
            return [
                {'type': 'text', 'text': 'You are first given the following demonstration video of how to perform the task. Use this as a guide.'},
                {'type': 'video', 'video': str(example)},
            ]
        return super()._get_examples_video(example)

    def _load_video(self, path):
        """Decode once per episode, preserving original frame timestamps.

        Match the reference VS sampler: uniform frames over the entire clip,
        capped at video_max_frames with one sample per video_interval seconds.
        Resize while decoding to bound the CPU cache for long tutorial videos.
        """
        if path not in self._video_cache:
            import numpy as np
            from decord import VideoReader, cpu
            reader = VideoReader(path, ctx=cpu(0), num_threads=1)
            total, fps = len(reader), float(reader.get_avg_fps())
            if total < 1 or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f'Cannot read video frames/FPS: {path}')
            count = min(max(round(total / fps / self.video_interval), self.video_min_frames), self.video_max_frames)
            indices = np.unique(np.linspace(0, total - 1, count, dtype=int)).tolist()
            # Native video processors require at least one temporal patch.
            if len(indices) == 1:
                indices *= 2
            height, width = reader[0].shape[:2]
            scale = min(1.0, math.sqrt(self.max_video_pixels / (height * width)))
            if scale < 1:
                reader = VideoReader(path, ctx=cpu(0), num_threads=1,
                                     width=max(1, int(width * scale)), height=max(1, int(height * scale)))
            frames = reader.get_batch(indices).asnumpy().transpose(0, 3, 1, 2)
            metadata = {'total_num_frames': total, 'fps': fps, 'duration': total / fps,
                        'frames_indices': indices, 'video_backend': 'decord'}
            self._video_cache[path] = (frames, metadata)
        return self._video_cache[path]

    def _local_inputs(self, messages):
        native_messages, images, videos, metadata = [], [], [], []
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                native_messages.append(dict(message))
                continue
            parts = []
            for part in content:
                if part['type'] == 'text':
                    parts.append({'type': 'text', 'text': part['text']})
                elif part['type'] == 'image_url':
                    url = part['image_url']['url']
                    if not url.startswith('data:image/') or ';base64,' not in url:
                        raise ValueError('Screenshots require base64 image URLs; use a native video part for tutorial videos')
                    with Image.open(BytesIO(base64.b64decode(url.split(';base64,', 1)[1]))) as image:
                        image = image.convert('RGB')
                    images.append(image)
                    parts.append({'type': 'image', 'image': image})
                elif part['type'] == 'video':
                    frames, info = self._load_video(part['video'])
                    videos.append(frames)
                    metadata.append({**info, 'frames_indices': list(info['frames_indices'])})
                    parts.append({'type': 'video', 'video': part['video']})
                else:
                    raise ValueError(f'Unsupported local message content: {part["type"]}')
            native_messages.append({'role': message['role'], 'content': parts})
        # Render via the tokenizer directly: older processor versions silently
        # discard enable_thinking. Preserve the real system and every turn.
        text = self.processor.tokenizer.apply_chat_template(
            native_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        kwargs = {'text': [text], 'return_tensors': 'pt'}
        if images:
            kwargs['images'] = resize_history_images(images, self.processor)
        if videos:
            kwargs['videos'] = videos
            # Qwen3-VL/Qwen3.5 expands video placeholders with frame timestamps.
            # Supply original FPS/indices and do not sample the cached clip again.
            kwargs['videos_kwargs'] = {
                'do_sample_frames': False,
                'video_metadata': metadata,
                'size': {'longest_edge': max(len(v) for v in videos) * self.max_video_pixels,
                         'shortest_edge': self.processor.video_processor.size['shortest_edge']},
            }
        inputs = dict(self.processor(**kwargs))
        # Same multimodal token-type convention as the existing VS runtime.
        # Keep this small bridge independent of the training dataset, which
        # otherwise injects its own system prompt and rewrites conversations.
        token_types = torch.zeros_like(inputs['input_ids'], dtype=torch.int32)
        for token, kind in [('<|image_pad|>', 1), ('<|video_pad|>', 2)]:
            token_types.masked_fill_(
                inputs['input_ids'] == self.processor.tokenizer.convert_tokens_to_ids(token), kind,
            )
        inputs['mm_token_type_ids'] = token_types
        return inputs

    def _eos_token_ids(self):
        """Stop at both checkpoint EOS and the tokenizer's chat-turn boundary."""
        checkpoint_eos = getattr(getattr(self.local_model, 'generation_config', None), 'eos_token_id', None)
        candidates = list(checkpoint_eos) if isinstance(checkpoint_eos, (list, tuple)) else [checkpoint_eos]
        tokenizer = self.processor.tokenizer
        candidates.append(tokenizer.eos_token_id)
        im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if im_end != tokenizer.unk_token_id:
            candidates.append(im_end)
        stop_ids = list(dict.fromkeys(token for token in candidates if isinstance(token, int) and token >= 0))
        if not stop_ids:
            raise ValueError('No valid checkpoint/tokenizer EOS token IDs found')
        return stop_ids

    def call_llm(self, payload, model):
        inputs = self._local_inputs(payload['messages'])
        inputs = {key: value.to(self.local_model.device) if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
        temperature = payload.get('temperature', self.temperature)
        generation = {
            'max_new_tokens': payload.get('max_tokens', self.max_tokens),
            'do_sample': temperature > 0,
            'eos_token_id': self._eos_token_ids(),
        }
        if temperature > 0:
            generation.update(temperature=temperature, top_p=payload.get('top_p', self.top_p))
        with torch.inference_mode():
            output = self.local_model.generate(**inputs, **generation)
        return self.processor.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )


def configure_processor(processor, data_args):
    """Apply the same processor limits as the existing VS dataset adapter."""
    image_processor = processor.image_processor
    image_processor.max_pixels = data_args.max_pixels
    image_processor.min_pixels = data_args.min_pixels
    image_processor.size['longest_edge'] = data_args.max_pixels
    image_processor.size['shortest_edge'] = data_args.min_pixels
