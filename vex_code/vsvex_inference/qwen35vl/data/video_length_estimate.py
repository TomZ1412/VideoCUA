"""Estimate visual tokens from container metadata, without decoding any frames."""

from concurrent.futures import Future
import io
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class TosRangeReader(io.RawIOBase):
    """Seekable, block-cached range reader for TorchCodec/FFmpeg metadata probing."""
    def __init__(self, client, key, block_size=256 * 1024):
        super().__init__()
        self.client, self.key, self.block_size = client, key, block_size
        self.size = int(self._retry(lambda: client.head_object(key)).headers["Content-Length"])
        self.position = 0
        self.blocks = {}

    @staticmethod
    def _retry(call):
        for attempt in range(3):
            try:
                return call()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size if whence == 2 else -1)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        parts = []
        while self.position < end:
            block = self.position // self.block_size
            start = block * self.block_size
            if block not in self.blocks:
                stop = min(self.size, start + self.block_size)
                response = self._retry(lambda: self.client.get_object_range(self.key, start, stop - 1))
                data = response.data
                if len(data) != stop - start:
                    raise ValueError(f"Incomplete TOS range response for {self.key}")
                self.blocks[block] = data
            data = self.blocks[block]
            count = min(end - self.position, len(data) - (self.position - start))
            parts.append(data[self.position - start:self.position - start + count])
            self.position += count
        return b"".join(parts)


class VideoTokenEstimator:
    def __init__(self, dataset, directory):
        self.dataset = dataset
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.futures = {}
        self.lock = threading.Lock()

    def probe(self, source):
        with self.lock:
            future = self.futures.get(source)
            owner = future is None
            if owner:
                future = self.futures[source] = Future()
        if owner:
            try:
                future.set_result(self._probe(source))
            except Exception as exc:
                future.set_exception(exc)
        return future.result()

    def _probe(self, source):
        from torchcodec.decoders import VideoDecoder
        field, location = source
        stat = Path(location).stat() if field == "video" else None
        key = digest({"version": 1, "source": source,
                      "stat": [stat.st_size, stat.st_mtime_ns] if stat else None})
        path = self.directory / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        reader = TosRangeReader(self.dataset.cli, location) if field == "tos_key" else None
        try:
            decoder = VideoDecoder(reader if reader is not None else location,
                                   seek_mode="approximate", num_ffmpeg_threads=1)
            meta = decoder.metadata
            duration, fps, frames = meta.duration_seconds, meta.average_fps, meta.num_frames
            if not fps or fps <= 0:
                fps = frames / duration if frames and duration else None
            if not frames and duration and fps:
                frames = max(1, round(duration * fps))
            if not frames or not fps or not meta.width or not meta.height:
                raise ValueError(f"Insufficient video metadata: {location}")
            result = {"num_frames": int(frames), "fps": float(fps),
                      "height": int(meta.height), "width": int(meta.width)}
            del decoder
        finally:
            if reader is not None:
                reader.close()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result))
        temporary.replace(path)
        return result

    def video_tokens(self, entry):
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
        args = self.dataset.data_args
        processor = args.video_processor
        field = "video" if "video" in entry else "tos_key"
        location = entry[field]
        if field == "video":
            location = os.path.abspath(location)
        meta = self.probe((field, location))
        frames, fps = meta["num_frames"], meta["fps"]
        duration = frames / fps
        interval = args.base_interval
        max_frames = args.video_max_frames
        pixels = args.video_max_frame_pixels
        sampled = min(max(round(duration / interval), args.video_min_frames), max_frames, frames)
        sampled = max(1, sampled)
        budget = max_frames * pixels
        factor = processor.patch_size * processor.merge_size
        if getattr(processor, "cap_pixels_per_frame", False):
            cap = processor.max_video_tokens * factor**2
            budget = max(min(cap, budget // sampled), int(args.video_min_frame_pixels * 1.05)) * sampled
        height, width = smart_resize(
            num_frames=max(sampled, processor.temporal_patch_size), height=meta["height"], width=meta["width"],
            temporal_factor=processor.temporal_patch_size, factor=factor,
            min_pixels=args.video_min_frame_pixels, max_pixels=budget,
        )
        return math.ceil(sampled / processor.temporal_patch_size) * (height // factor) * (width // factor)

