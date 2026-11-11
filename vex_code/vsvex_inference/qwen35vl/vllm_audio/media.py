"""Decode raw HTTP video bytes using the same Dataset as offline inference."""
from collections import OrderedDict
from concurrent.futures import Future
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time

import numpy as np


logger = logging.getLogger('vllm.qwenvl_audio.media')
_datasets = threading.local()


def _bundle_size(value):
    """Account for tensor storage as well as Python token lists (CPU only)."""
    if hasattr(value, 'untyped_storage'):
        return value.untyped_storage().nbytes()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(_bundle_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(_bundle_size(v) for v in value)
    return sys.getsizeof(value)


class BundleCache:
    """Byte-bounded LRU with one producer for concurrent requests for a video.

    Cached tensors are read-only: AVProcessor copies the bundle dictionary
    before removing fields and never modifies the cached tensor storage.
    Oversized results are shared with current waiters but are not retained.
    """
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.size = 0
        self.entries = OrderedDict()
        self.pending = {}
        self.lock = threading.Lock()

    def get_or_load(self, key, load):
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return self.entries[key][0], 'hit'
            future = self.pending.get(key)
            owner = future is None
            if owner:
                future = self.pending[key] = Future()
        if not owner:
            return future.result(), 'shared'
        try:
            value = load()
            size = _bundle_size(value)
            with self.lock:
                if 0 < size <= self.max_bytes:
                    while self.size + size > self.max_bytes:
                        _, (_, removed_size) = self.entries.popitem(last=False)
                        self.size -= removed_size
                    self.entries[key] = (value, size)
                    self.size += size
                future.set_result(value)
                del self.pending[key]
            return value, 'miss'
        except BaseException as exc:
            with self.lock:
                future.set_exception(exc)
                del self.pending[key]
            raise


_cache = None
_cache_lock = threading.Lock()


def _get_cache():
    global _cache
    with _cache_lock:
        if _cache is None:
            budget = float(os.environ.get('QWENVL_MEDIA_CACHE_GB', '4'))
            if not np.isfinite(budget) or budget < 0:
                raise ValueError('QWENVL_MEDIA_CACHE_GB must be finite and nonnegative')
            _cache = BundleCache(int(budget * 1024 ** 3))
        return _cache


def _decode_bundle(data, model, options, use_audio):
    from .data import make_dataset, pack_item
    # Retain one Dataset/processor set per media worker. Never share mutable
    # tokenizer/processor state between threads, or reload it for every question.
    key = (model, json.dumps(options, sort_keys=True))
    with tempfile.NamedTemporaryFile(suffix='.mp4') as handle:
        handle.write(data)
        handle.flush()
        sample = dict(video=handle.name, use_audio=use_audio,
                      conversations=[{'from': 'human', 'value': 'AVBEGIN<video>AVEND'}])
        if getattr(_datasets, 'key', None) != key:
            _datasets.dataset = make_dataset(model, [sample], options)
            _datasets.key = key
        dataset = _datasets.dataset
        dataset.list_data_dict[0] = sample
        item = dataset[0]
        bundle = pack_item(item)
        text = dataset.tokenizer.decode(item['input_ids'][0], skip_special_tokens=False)
        media_text = text.split('AVBEGIN', 1)[1].split('AVEND', 1)[0]
        bundle['av_media_ids'] = dataset.tokenizer.encode(media_text, add_special_tokens=False)
    return bundle


def load_bundle(data, model, options, use_audio):
    started = time.monotonic()
    # Hash bytes rather than URL/tempfile name: changed content cannot reuse old
    # features, and identical HDFS/TOS/HTTP/local videos can share preprocessing.
    key = (model, json.dumps(options, sort_keys=True), bool(use_audio),
           hashlib.sha256(data).digest())
    bundle, status = _get_cache().get_or_load(
        key, lambda: _decode_bundle(data, model, options, use_audio))
    if os.environ.get('QWENVL_MEDIA_PROFILE') == '1':
        logger.info('AV media: cache=%s bytes=%s seconds=%.3f',
                    status, len(data), time.monotonic() - started)
    # Containers belong to each request; large immutable tensors share storage.
    return {k: list(v) if isinstance(v, list) else v for k, v in bundle.items()}


def register_video_loader():
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader
    if 'qwen35_audio' in VIDEO_LOADER_REGISTRY.name2class:
        return

    @VIDEO_LOADER_REGISTRY.register('qwen35_audio')
    class AudioVisualVideoLoader(VideoLoader):
        @classmethod
        def load_bytes(cls, data, num_frames=-1, **kwargs):
            model = os.environ['QWENVL_MODEL']
            options = json.loads(os.environ.get('QWENVL_MEDIA_OPTIONS', '{}'))
            use_audio = os.environ.get('QWENVL_USE_AUDIO', '1') == '1' and kwargs.get('use_audio', True)
            bundle = load_bundle(data, model, options, use_audio)
            # The custom parser consumes the tensors in metadata, not these frames.
            return np.zeros((1, 32, 32, 3), dtype=np.uint8), {'qwenvl_bundle': bundle}
