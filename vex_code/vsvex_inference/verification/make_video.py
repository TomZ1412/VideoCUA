import json
import os
import subprocess
from tqdm import tqdm
import time


def get_audio_duration(audio_path):
    """Use ffprobe to get the exact duration of an audio file."""

    if not audio_path:
        raise RuntimeError("Audio path is empty.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file does not exist: {audio_path}"
        )

    if os.path.getsize(audio_path) == 0:
        raise RuntimeError(
            f"Audio file is empty: {audio_path}"
        )

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for audio:\n"
            f"  {audio_path}\n"
            f"stderr:\n"
            f"{result.stderr.strip()}"
        )

    duration_str = result.stdout.strip()

    if not duration_str:
        raise RuntimeError(
            f"ffprobe returned empty duration for: {audio_path}"
        )

    try:
        duration = float(duration_str)
    except ValueError:
        raise RuntimeError(
            f"Invalid duration '{duration_str}' "
            f"for audio: {audio_path}"
        )

    if duration <= 0:
        raise RuntimeError(
            f"Invalid audio duration {duration} "
            f"for: {audio_path}"
        )

    return duration


def create_video_ffmpeg_overlap(
    image_list,
    audio_list,
    output_name,
):
    """
    Create video from screenshots and audio.

    Returns:
        True  -> success
        False -> failed
    """

    has_extra_image = len(image_list) > len(audio_list)

    temp_dir = "temp_overlap4"
    os.makedirs(temp_dir, exist_ok=True)

    # Important:
    # initialize first so finally will never reference
    # an undefined local variable.
    video_list_path = None
    audio_list_path = None
    master_audio = None
    master_video = None

    try:
        N = len(audio_list)

        if N == 0:
            raise RuntimeError("audio_list is empty")

        if len(image_list) < N:
            raise RuntimeError(
                f"Not enough images: "
                f"{len(image_list)} images vs {N} audios"
            )

        video_list_path = os.path.join(
            temp_dir,
            "video_list.txt",
        )

        audio_list_path = os.path.join(
            temp_dir,
            "audio_list.txt",
        )

        # ----------------------------------------------------------
        # 1. Probe audio durations and construct concat files
        # ----------------------------------------------------------
        with open(video_list_path, "w") as vf, \
             open(audio_list_path, "w") as af:

            for i in range(N):

                audio_path = audio_list[i]
                image_path = image_list[i]

                print(
                    f"[Audio {i + 1}/{N}] "
                    f"{audio_path}"
                )

                # Check image
                if not image_path:
                    raise RuntimeError(
                        f"Image path is empty at index {i}"
                    )

                if not os.path.exists(image_path):
                    raise FileNotFoundError(
                        f"Image does not exist: {image_path}"
                    )

                # Check/probe audio
                dur = get_audio_duration(audio_path)

                # Image duration is 0.5 s shorter than audio.
                # Keep at least 0.1 s.
                vid_dur = max(
                    0.1,
                    dur - 0.5,
                )

                img_path_abs = (
                    os.path.abspath(image_path)
                    .replace("\\", "/")
                )

                aud_path_abs = (
                    os.path.abspath(audio_path)
                    .replace("\\", "/")
                )

                vf.write(
                    f"file '{img_path_abs}'\n"
                    f"duration {vid_dur}\n"
                )

                af.write(
                    f"file '{aud_path_abs}'\n"
                )

            # ------------------------------------------------------
            # Extra final screenshot
            # ------------------------------------------------------
            if has_extra_image:

                last_img_path = image_list[-1]

                if not last_img_path:
                    raise RuntimeError(
                        "Last image path is empty"
                    )

                if not os.path.exists(last_img_path):
                    raise FileNotFoundError(
                        f"Last image does not exist: "
                        f"{last_img_path}"
                    )

                # Each screenshot was shortened by 0.5 seconds.
                # Add all removed time back to the final screenshot
                # plus a 3 s pause.
                last_img_dur = (
                    3.0 + 0.5 * N
                )

                last_img_path_abs = (
                    os.path.abspath(last_img_path)
                    .replace("\\", "/")
                )

                vf.write(
                    f"file '{last_img_path_abs}'\n"
                    f"duration {last_img_dur}\n"
                )

        # ----------------------------------------------------------
        # 2. Create master audio
        # ----------------------------------------------------------
        master_audio = os.path.join(
            temp_dir,
            "master_audio.aac",
        )

        cmd_audio = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", audio_list_path,
            "-c:a", "aac",
            "-ar", "16000",
            "-b:a", "128k",
            master_audio,
        ]

        result_audio = subprocess.run(
            cmd_audio,
            capture_output=True,
            text=True,
        )

        if result_audio.returncode != 0:
            raise RuntimeError(
                "Failed to create master audio.\n"
                f"stderr:\n"
                f"{result_audio.stderr.strip()}"
            )

        # ----------------------------------------------------------
        # 3. Create master video
        # ----------------------------------------------------------
        master_video = os.path.join(
            temp_dir,
            "master_video.mp4",
        )

        cmd_video = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", video_list_path,
            "-vf", "scale=-2:720",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-r", "8",
            master_video,
        ]

        result_video = subprocess.run(
            cmd_video,
            capture_output=True,
            text=True,
        )

        if result_video.returncode != 0:
            raise RuntimeError(
                "Failed to create master video.\n"
                f"stderr:\n"
                f"{result_video.stderr.strip()}"
            )

        # ----------------------------------------------------------
        # 4. Mux audio + video
        # ----------------------------------------------------------
        cmd_mux = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", master_video,
            "-i", master_audio,
            "-c", "copy",
            "-map", "0:v",
            "-map", "1:a",
            output_name,
        ]

        result_mux = subprocess.run(
            cmd_mux,
            capture_output=True,
            text=True,
        )

        if result_mux.returncode != 0:
            raise RuntimeError(
                "Failed to mux audio and video.\n"
                f"stderr:\n"
                f"{result_mux.stderr.strip()}"
            )

        # Double check output
        if not os.path.exists(output_name):
            raise RuntimeError(
                f"Output video was not created: "
                f"{output_name}"
            )

        if os.path.getsize(output_name) == 0:
            raise RuntimeError(
                f"Output video is empty: "
                f"{output_name}"
            )

        return True

    except Exception as e:

        print()
        print("=" * 80)
        print(
            f"Failed to create video:"
        )
        print(
            f"  {output_name}"
        )
        print(
            f"Reason:"
        )
        print(
            f"  {e}"
        )
        print("=" * 80)

        # In case ffmpeg created a partial/broken output
        if os.path.exists(output_name):
            try:
                os.remove(output_name)
            except OSError:
                pass

        return False

    finally:

        # ----------------------------------------------------------
        # Clean temporary files safely
        # ----------------------------------------------------------
        temp_files = [
            video_list_path,
            audio_list_path,
            master_audio,
            master_video,
        ]

        for f in temp_files:

            if f is None:
                continue

            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

        if (
            os.path.exists(temp_dir)
            and os.path.isdir(temp_dir)
        ):
            try:
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except OSError:
                pass


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":

    # --------------------------------------------------------------
    # Config
    # --------------------------------------------------------------
    input_json = (
        os.environ.get("VEX_INPUT_JSON", 'YOUR_AUDIO_TRAJECTORIES_PATH')
    )

    output_json = (
        __import__("os").environ.get("VEX_OUTPUT_JSON", 'YOUR_VIDEO_TRAJECTORIES_PATH')
    )

    video_dir = (
        os.environ.get("VEX_VIDEO_DIR", 'YOUR_VIDEO_OUTPUT_PATH')
    )

    os.makedirs(
        video_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------
    with open(
        input_json,
        "r",
    ) as fin:

        data = json.load(fin)

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------
    success_count = 0
    existing_count = 0
    failed_count = 0
    skipped_no_steps = 0
    skipped_missing_audio_field = 0
    skipped_bad_audio_file = 0

    failed_tasks = []

    # --------------------------------------------------------------
    # Generate videos
    # --------------------------------------------------------------
    for datapiece in tqdm(
        data,
        desc="Generating Videos",
    ):

        task_id = datapiece.get(
            "task",
            "UNKNOWN_TASK",
        )

        steps = datapiece.get(
            "steps",
            [],
        )

        # ----------------------------------------------------------
        # No steps
        # ----------------------------------------------------------
        if not steps:

            print(
                f"Skip {task_id}: "
                f"no steps"
            )

            skipped_no_steps += 1

            continue

        # ----------------------------------------------------------
        # Check that every step has an audio field
        # ----------------------------------------------------------
        if any(
            "audio" not in step
            for step in steps
        ):

            print(
                f"Skip {task_id}: "
                f"some steps have no audio field"
            )

            skipped_missing_audio_field += 1

            continue

        # ----------------------------------------------------------
        # Check actual audio files
        # ----------------------------------------------------------
        bad_audio = None

        for i, step in enumerate(steps):

            audio_path = step.get(
                "audio"
            )

            if not audio_path:

                bad_audio = (
                    i,
                    audio_path,
                    "empty audio path",
                )

                break

            if not os.path.exists(
                audio_path
            ):

                bad_audio = (
                    i,
                    audio_path,
                    "file does not exist",
                )

                break

            if os.path.getsize(
                audio_path
            ) == 0:

                bad_audio = (
                    i,
                    audio_path,
                    "file is empty",
                )

                break

        if bad_audio is not None:

            step_idx, audio_path, reason = (
                bad_audio
            )

            print(
                f"Skip {task_id}: "
                f"bad audio at step {step_idx}: "
                f"{audio_path} "
                f"({reason})"
            )

            skipped_bad_audio_file += 1

            failed_tasks.append(
                {
                    "task": task_id,
                    "reason": reason,
                    "step_idx": step_idx,
                    "audio": audio_path,
                }
            )

            continue

        # ----------------------------------------------------------
        # Build image/audio lists
        # ----------------------------------------------------------
        imagelist = [
            step["screenshot"]
            for step in steps
        ]

        audiolist = [
            step["audio"]
            for step in steps
        ]

        init_screenshot = datapiece.get(
            "init_screenshot"
        )

        if not init_screenshot:

            print(
                f"Skip {task_id}: "
                f"missing init_screenshot"
            )

            failed_count += 1

            failed_tasks.append(
                {
                    "task": task_id,
                    "reason": "missing init_screenshot",
                }
            )

            continue

        imagelist = [
            init_screenshot
        ] + imagelist

        output_path = os.path.join(
            video_dir,
            f"{task_id}.mp4",
        )

        # ----------------------------------------------------------
        # Resume support
        # ----------------------------------------------------------
        if (
            os.path.exists(output_path)
            and os.path.getsize(output_path) > 0
        ):

            datapiece["video"] = (
                output_path
            )

            existing_count += 1

            continue

        # Remove empty/corrupted old output
        if os.path.exists(output_path):

            try:
                os.remove(output_path)
            except OSError:
                pass

        # ----------------------------------------------------------
        # Generate
        # ----------------------------------------------------------
        success = (
            create_video_ffmpeg_overlap(
                imagelist,
                audiolist,
                output_path,
            )
        )

        if success:

            datapiece["video"] = (
                output_path
            )

            success_count += 1

        else:

            print(
                f"Skip failed task: "
                f"{task_id}"
            )

            failed_count += 1

            failed_tasks.append(
                {
                    "task": task_id,
                    "reason": (
                        "video generation failed"
                    ),
                }
            )

        # ----------------------------------------------------------
        # Incremental save
        #
        # Important for long jobs:
        # if the process is killed, previous progress is preserved.
        # ----------------------------------------------------------
        with open(
            output_json,
            "w",
        ) as fout:

            json.dump(
                data,
                fout,
                indent=4,
                ensure_ascii=False,
            )

    # --------------------------------------------------------------
    # Final save
    # --------------------------------------------------------------
    with open(
        output_json,
        "w",
    ) as fout:

        json.dump(
            data,
            fout,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------------
    # Save failed task information
    # --------------------------------------------------------------
    failed_json = (
        'YOUR_VIDEO_FAILURE_LOG_PATH'
    )

    with open(
        failed_json,
        "w",
    ) as fout:

        json.dump(
            failed_tasks,
            fout,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)

    print(
        f"New videos generated: "
        f"{success_count}"
    )

    print(
        f"Existing videos skipped: "
        f"{existing_count}"
    )

    print(
        f"Video generation failures: "
        f"{failed_count}"
    )

    print(
        f"No-step tasks skipped: "
        f"{skipped_no_steps}"
    )

    print(
        f"Missing audio-field tasks skipped: "
        f"{skipped_missing_audio_field}"
    )

    print(
        f"Bad/missing audio-file tasks skipped: "
        f"{skipped_bad_audio_file}"
    )

    print(
        f"Output JSON: "
        f"{output_json}"
    )

    print(
        f"Failed task log: "
        f"{failed_json}"
    )