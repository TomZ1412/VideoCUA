"""Qwen3.5 dense/MoE with the checkpoint's Whisper + Q-Former branch.

The language/vision kernels and hybrid attention remain native vLLM. Audio
weights are replicated across TP ranks; audio chunks are encoded one at a time
so peak encoder memory does not grow with video duration.
"""
import itertools
import torch
from transformers import BatchFeature
from vllm.inputs import mm_input
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems, PlaceholderRange
from vllm.multimodal.parse import MultiModalDataParser, ProcessorBatchItems
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo, Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.utils import WeightsMapper
from qwenvl.model.configuration_audio import AudioWhisperConfig
from qwenvl.model.modeling_whisper import WhisperEncoder
from .cache import limit_shm_object_size


logger = init_logger('vllm.qwenvl_audio')


class AVParser(MultiModalDataParser):
    def _parse_video_data(self, data):
        if isinstance(data, dict):
            items = [data]
        else:
            # vLLM's HTTP media loader returns (frames, metadata).
            if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
                data = [data]
            items = []
            for value in data:
                if isinstance(value, dict):
                    items.append(value)
                elif isinstance(value, tuple) and 'qwenvl_bundle' in value[1]:
                    items.append(value[1]['qwenvl_bundle'])
                else:
                    raise ValueError('Use the qwen35_audio video loader or a preprocessed Dataset bundle')
        if len(items) != 1:
            raise ValueError('Supply one video bundle per request (offline bundles may contain multiple clips)')
        return ProcessorBatchItems(items, 'video')


class AVInfo(Qwen3_5ProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_data_parser(self):
        return AVParser()

    def get_supported_mm_limits(self):
        return {'video': 1}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        # Audio duration is independent of sampled frame count. The sequence
        # budget is the safe upper bound for the fused AV bundle.
        return {'video': seq_len}


class AVDummy(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return '<|vision_start|><|video_pad|><|vision_end|>' * mm_counts.get('video', 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        if not mm_counts.get('video', 0):
            return {}
        cfg = self.info.get_hf_config()
        # Profile a context-sized AV input rather than a tiny placeholder.
        # Vision dominates transient memory; audio is already encoded one chunk
        # at a time. Keep one real-sized audio chunk while filling the remaining
        # token budget with visual patches. This intentionally uses the context
        # ceiling, rather than assuming every real video exhausts a frame budget.
        v = cfg.vision_config
        audio_cfg = cfg.audio_config
        audio_cfg = AudioWhisperConfig(**audio_cfg) if isinstance(audio_cfg, dict) else audio_cfg
        window = round(1500 * audio_cfg.second_per_window / 30)
        stride = round(1500 * audio_cfg.second_stride / 30)
        audio_tokens = ((1500 - window) // stride + 1) * audio_cfg.qformer_config.query_length
        visual_tokens = seq_len - audio_tokens - 2
        if visual_tokens < 1:
            raise ValueError('max_model_len is too small to profile one audio-visual input')
        merge = v.spatial_merge_size
        # Each grid row describes complete temporal patches. Full rows use a
        # 16x16 grid after spatial merging; the final row fills the remainder.
        frames, remainder = divmod(visual_tokens, 256)
        grids = ([[frames, 16 * merge, 16 * merge]] if frames else [])
        if remainder:
            grids.append([1, merge, remainder * merge])
        ids = ([cfg.vision_start_token_id] + [cfg.video_token_id] * visual_tokens
               + [cfg.audio_token_id] * audio_tokens + [cfg.vision_end_token_id])
        logger.info('AV profiling input: visual_tokens=%s, audio_tokens=%s, grid_thw=%s',
                    visual_tokens, audio_tokens, grids)
        return {'video': dict(
            pixel_values_videos=torch.zeros(
                visual_tokens * merge ** 2,
                3 * v.temporal_patch_size * v.patch_size ** 2),
            video_grid_thw=torch.tensor(grids),
            audio_feature=torch.zeros(1, 128, 3000),
            av_token_ids=torch.tensor(ids), av_media_ids=ids)}


class AVProcessor(Qwen3VLMultiModalProcessor):
    def apply(self, inputs, timing_ctx):
        tokenizer = self.info.get_tokenizer()
        prompt = inputs.prompt
        items = inputs.mm_data_items
        if not items.get_count('video', strict=False):
            ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
            return mm_input(prompt_token_ids=ids, mm_kwargs=MultiModalKwargsItems({}),
                            mm_hashes={}, mm_placeholders={})
        bundle = dict(items['video'].get(0))
        if 'av_media_ids' in bundle:
            text = prompt if isinstance(prompt, str) else tokenizer.decode(prompt, skip_special_tokens=False)
            marker = '<|vision_start|><|video_pad|><|vision_end|>'
            if text.count(marker) != 1:
                raise ValueError('Expected exactly one video placeholder in the chat prompt')
            text = text.replace(marker, tokenizer.decode(bundle.pop('av_media_ids'), skip_special_tokens=False), 1)
            ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
            if ids != bundle['av_token_ids'].tolist():
                raise ValueError('Preprocessed prompt and Dataset token IDs differ')
        cfg = self.info.get_hf_config()
        tokens = torch.tensor(ids, dtype=torch.long)
        mask = (tokens == cfg.video_token_id) | (tokens == cfg.image_token_id) | (tokens == cfg.audio_token_id)
        indices = mask.nonzero().flatten()
        if not len(indices):
            raise ValueError('Multimodal input contains no embedding placeholders')
        for pixel_key, grid_key, token_id in (
            ('pixel_values_videos', 'video_grid_thw', cfg.video_token_id),
            ('pixel_values', 'image_grid_thw', cfg.image_token_id),
        ):
            expected = (int(bundle[grid_key].prod(-1).sum()) // cfg.vision_config.spatial_merge_size ** 2
                        if grid_key in bundle else 0)
            if int((tokens == token_id).sum()) != expected or (expected and pixel_key not in bundle):
                raise ValueError(f'{pixel_key}: token count and visual features disagree')
        audio_cfg = cfg.audio_config
        audio_cfg = AudioWhisperConfig(**audio_cfg) if isinstance(audio_cfg, dict) else audio_cfg
        window = round(1500 * audio_cfg.second_per_window / 30)
        stride = round(1500 * audio_cfg.second_stride / 30)
        per_chunk = ((1500 - window) // stride + 1) * audio_cfg.qformer_config.query_length
        chunks = bundle['audio_feature'].shape[0] if 'audio_feature' in bundle else 0
        if int((tokens == cfg.audio_token_id).sum()) != chunks * per_chunk:
            raise ValueError('Audio token count and features disagree')
        start, end = int(indices[0]), int(indices[-1]) + 1
        # av_order only describes embedding positions. Timestamp/text positions
        # are excluded, and vLLM uses the span mask during chunked prefill.
        bundle.pop('av_token_ids')
        bundle['av_order'] = tokens[mask]
        data = {key: [value] for key, value in bundle.items()}
        fields = {key: MultiModalFieldConfig.batched('video', keep_on_cpu=key.endswith('grid_thw'))
                  for key in data}
        kwargs = MultiModalKwargsItems.from_hf_inputs(BatchFeature(data), fields)
        mm_hashes = inputs.get_mm_hashes(self.info.ctx.model_config.model)
        # Our custom apply() bypasses the base processor's cache merge. Without
        # this handoff, multi-GB AV tensors are pickled and copied to every TP
        # worker for every question, even when their GPU prefix is cached.
        # Use the same sender/receiver protocol as the native processor. Prompt
        # IDs and placeholder ranges remain request-specific; only features are
        # shared. SHM caches return an address which workers resolve before use.
        if self.cache is not None:
            limit_shm_object_size(self.cache)
            item_hash = mm_hashes['video'][0]
            self.cache.touch_sender_cache_item(item_hash)
            cached, _ = self.cache.get_and_update_item((kwargs['video'][0], []), item_hash)
            kwargs = MultiModalKwargsItems({'video': [cached]})
        return mm_input(prompt_token_ids=ids, mm_kwargs=kwargs,
                        mm_hashes=mm_hashes,
                        mm_placeholders={'video': [PlaceholderRange(start, end-start, mask[start:end])]})


def mrope_positions(input_tokens, grids, config):
    """Match Transformers Qwen3.5: audio pads are 1D text positions."""
    tokens = torch.tensor(input_tokens)
    types = torch.zeros_like(tokens)
    types[tokens == config.image_token_id] = 1
    types[tokens == config.video_token_id] = 2
    grid_iters = {}
    for kind, key in ((1, 'image_grid_thw'), (2, 'video_grid_thw')):
        rows = grids.get(key, [])
        frames = []
        for row in rows:
            t, h, w = map(int, row)
            frames.extend([(1, h, w)] * t)
        grid_iters[kind] = iter(frames)
    result, current = [], 0
    merge = config.vision_config.spatial_merge_size
    for kind, group in itertools.groupby(types.tolist()):
        count = sum(1 for _ in group)
        if kind == 0:
            result.append(torch.arange(count).expand(3, -1) + current)
            current += count
        else:
            _, h, w = next(grid_iters[kind])
            h, w = h // merge, w // merge
            if count != h * w:
                raise ValueError(f'Visual token run {count} does not match grid {h}x{w}')
            result.append(torch.stack((torch.zeros(h*w, dtype=torch.long),
                                      torch.arange(h).repeat_interleave(w),
                                      torch.arange(w).repeat(h))) + current)
            current += max(h, w)
    positions = torch.cat(result, dim=1) if result else torch.empty(3, 0, dtype=torch.long)
    return positions, current - len(input_tokens)


class AudioMixin:
    hf_to_vllm_mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={'model.audio.': 'audio.'})

    def __init__(self, *, vllm_config, prefix='model'):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg = self.config.audio_config
        if isinstance(cfg, dict):
            cfg = AudioWhisperConfig(**cfg)
        cfg._attn_implementation = 'sdpa'
        with self._mark_tower_model(vllm_config, {'video'}):
            self.audio = WhisperEncoder(cfg).eval()

    def embed_multimodal(self, **kwargs):
        if 'av_order' not in kwargs:
            return []
        orders = kwargs['av_order']
        outputs = []
        for i, order in enumerate(orders):
            order = order.flatten()
            embeddings = torch.empty(len(order), self.config.text_config.hidden_size,
                                     device=order.device, dtype=self.visual.dtype)
            for key, grid_key, token_id in (
                ('pixel_values_videos', 'video_grid_thw', self.config.video_token_id),
                ('pixel_values', 'image_grid_thw', self.config.image_token_id),
            ):
                if key in kwargs:
                    pixels, grid = kwargs[key][i], kwargs[grid_key][i]
                    # Capture the allocation baseline before ViT, so an OOM
                    # identifies the actual input and retained GPU memory.
                    memory_before = None
                    if pixels.is_cuda:
                        free, total = torch.cuda.mem_get_info(pixels.device)
                        memory_before = dict(
                            free_bytes=free, total_bytes=total,
                            allocated_bytes=torch.cuda.memory_allocated(pixels.device),
                            reserved_bytes=torch.cuda.memory_reserved(pixels.device))
                    try:
                        visual = self.visual(pixels.to(self.visual.dtype), grid_thw=grid)
                    except torch.OutOfMemoryError:
                        logger.exception(
                            'AV visual encoder OOM: item=%s/%s, modality=%s, '
                            'grid_thw=%s, pixels_shape=%s, pixels_dtype=%s, '
                            'visual_dtype=%s, embedding_tokens=%s, '
                            'audio_shape=%s, memory_before=%s',
                            i, len(orders), key, grid.tolist(), tuple(pixels.shape),
                            pixels.dtype, self.visual.dtype, len(order),
                            tuple(kwargs['audio_feature'][i].shape)
                            if 'audio_feature' in kwargs else None, memory_before)
                        raise
                    if visual.shape[0] != int((order == token_id).sum()):
                        raise ValueError('Visual encoder/placeholder count mismatch')
                    embeddings[order == token_id] = visual
            if 'audio_feature' in kwargs:
                features = kwargs['audio_feature'][i]
                audio = torch.cat([self.audio(chunk.unsqueeze(0).to(self.audio.dtype)).flatten(0, 1)
                                   for chunk in features], dim=0)
                if audio.shape[0] != int((order == self.config.audio_token_id).sum()):
                    raise ValueError('Audio encoder/placeholder count mismatch')
                embeddings[order == self.config.audio_token_id] = audio.to(embeddings.dtype)
            outputs.append(embeddings)
        return outputs

    def get_mrope_input_positions(self, input_tokens, mm_features):
        grids = {}
        for feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            for key in ('image_grid_thw', 'video_grid_thw'):
                if key in feature.data:
                    grids.setdefault(key, []).extend(feature.data[key].data.reshape(-1, 3).tolist())
        return mrope_positions(input_tokens, grids, self.config)


@MULTIMODAL_REGISTRY.register_processor(AVProcessor, info=AVInfo, dummy_inputs=AVDummy)
class Qwen3_5AudioForConditionalGeneration(AudioMixin, Qwen3_5ForConditionalGeneration):
    pass


@MULTIMODAL_REGISTRY.register_processor(AVProcessor, info=AVInfo, dummy_inputs=AVDummy)
class Qwen3_5MoeAudioForConditionalGeneration(AudioMixin, Qwen3_5MoeForConditionalGeneration):
    pass
