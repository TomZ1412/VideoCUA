import base64
from io import BytesIO
from typing import Tuple

from PIL import Image

from mm_agents.utils.qwen_vl_utils import smart_resize
import os
import tempfile
import subprocess
import base64
import json
import cv2

TARGET_HEIGHT = 360
TARGET_FPS = 1
MAX_FRAMES = 256
JPEG_QUALITY = 90


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

def extract_frames(video_path):
    """Sample the video at 1 FPS, capped at MAX_FRAMES frames.
    If the video is too long for 1 FPS, frames are evenly spaced
    across the whole video. Returns (frames, duration, sample_fps)
    where frames is a list of (timestamp_sec, data_url)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0 or n_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read video FPS / frame count")

    duration = n_frames / src_fps

    # --- decide sampling schedule ---
    n_at_1fps = int(duration * TARGET_FPS)
    if n_at_1fps <= MAX_FRAMES:
        # short enough: keep 1 FPS
        sample_times = [i / TARGET_FPS for i in range(max(1, n_at_1fps))]
    else:
        # too long: MAX_FRAMES evenly spaced across the whole video
        sample_times = [i * duration / MAX_FRAMES for i in range(MAX_FRAMES)]

    sample_fps = len(sample_times) / duration if duration > 0 else 0.0

    frames = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                min(int(t * src_fps), n_frames - 1))
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 360p (height=360), keep aspect ratio
        h, w = frame.shape[:2]
        new_w = int(round(w * TARGET_HEIGHT / h))
        new_w += new_w % 2
        if (h, w) != (TARGET_HEIGHT, new_w):
            frame = cv2.resize(frame, (new_w, TARGET_HEIGHT),
                               interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode("utf-8")
        frames.append((t, f"data:image/jpeg;base64,{b64}"))

    cap.release()
    return frames, duration, sample_fps

def fmt_time(t):
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def build_content(frames, transcriptions, sample_fps):
    content = [{
        "type": "text",
        "text": (
            "You are first given the following demonstration video of how to perform "
            "a task that you will be performing next. You can use it as a guide or "
            "just follow this demonstration. The video is provided as frames sampled "
            f"at {sample_fps:.2f} FPS (360p), with the audio transcribed and "
            "interleaved at the matching timestamps."
        ),
    }]

    events = [(t, 0, {"type": "image_url", "image_url": {"url": url}})
              for t, url in frames]
    events += [(start, 1, {
        "type": "text",
        "text": f"[Audio {fmt_time(start)}-{fmt_time(end)}] {text}",
    }) for text, start, end in transcriptions]

    events.sort(key=lambda e: (e[0], e[1]))
    content.extend(item for _, _, item in events)
    return content


def process_image(image_bytes: bytes, max_pixels=16 * 16 * 4 * 12800) -> str:
    """Resize + re-encode screenshot and return base64 PNG."""
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=max_pixels,
    )

    image = image.resize((resized_width, resized_height))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_size_from_bytes(image_bytes: bytes) -> Tuple[int, int]:
    image = Image.open(BytesIO(image_bytes))
    return image.size


def image_size_from_base64(image_b64: str) -> Tuple[int, int]:
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    return image.size


def adjust_coordinates(
    x: float,
    y: float,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[int, int]:
    if not (original_width and original_height):
        return int(x), int(y)
    if coordinate_type == "absolute":
        if processed_width and processed_height:
            x_scale = original_width / processed_width
            y_scale = original_height / processed_height
            return int(x * x_scale), int(y * y_scale)
        return int(x), int(y)

    x_scale = original_width / 999
    y_scale = original_height / 999
    return int(x * x_scale), int(y * y_scale)
