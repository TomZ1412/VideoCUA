import base64
from io import BytesIO
from typing import Tuple
import os
import tempfile
import subprocess

from PIL import Image

from mm_agents.utils.qwen_vl_utils import smart_resize


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


def process_image_resize(image_bytes: bytes, max_pixels=16 * 16 * 4 * 12800) -> str:
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


def process_image(image_bytes: bytes) -> str:
    """Re-encode screenshot as base64 PNG without Qwen-specific resizing."""
    image = Image.open(BytesIO(image_bytes))
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
