import soundfile as sf
import soxr
from tqdm import tqdm
import os
import json
import signal
import time
import re
import tempfile
import numpy as np


# =========================
# Timeout
# =========================

class GenerationTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise GenerationTimeout("Audio generation timed out.")




# =========================
# Audio generation
# =========================

DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
NUMBER_PATTERN = re.compile(r"[0-9](?:[0-9,]*[0-9])?")


def prepare_tts_text(text, spell_all_numbers=False):
    """Keep source descriptions intact; normalize only text sent to Kokoro."""
    def replace_number(match):
        digits = match.group().replace(",", "")
        if len(digits) < 16 and not spell_all_numbers:
            return match.group()
        if len(digits) <= 64:
            spoken = " ".join(DIGIT_WORDS[int(d)] for d in digits)
        else:
            prefix = " ".join(DIGIT_WORDS[int(d)] for d in digits[:12])
            suffix = " ".join(DIGIT_WORDS[int(d)] for d in digits[-8:])
            spoken = (f"a numeric string of {len(digits)} digits, "
                      f"beginning with {prefix}, and ending with {suffix}")
        # Numeric strings can be embedded directly in a title or filename.
        return " " + spoken + " "

    return NUMBER_PATTERN.sub(replace_number, text)


def generate(text, pipeline, outpath):
    def synthesize(spoken_text):
        chunks = []
        for gs, ps, audio in pipeline(spoken_text, voice='af_heart'):
            if audio is not None and len(audio):
                chunks.append(soxr.resample(audio, 24000, 16000))
        if not chunks:
            raise ValueError("Kokoro produced no audio")
        return np.concatenate(chunks)

    spoken_text = prepare_tts_text(text)
    try:
        audio_16k = synthesize(spoken_text)
    except OverflowError:
        # Handle numeric formats that G2P still combines into oversized values.
        spoken_text = prepare_tts_text(text, spell_all_numbers=True)
        audio_16k = synthesize(spoken_text)

    # Only publish a complete WAV. A timeout/error must not leave a file that
    # the next run mistakes for an already completed step.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".kokoro_", suffix=".wav",
            dir=os.path.dirname(os.path.abspath(outpath)), delete=False,
        ) as temp:
            temp_path = temp.name
        sf.write(temp_path, audio_16k, 16000)
        os.replace(temp_path, outpath)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)
    return spoken_text


# =========================
# Paths
# =========================



traj_file = __import__("os").environ.get("VEX_TRAJ_FILE", 'YOUR_TRAJECTORIES_PATH')
traj_output = __import__("os").environ.get("VEX_TRAJ_OUTPUT", 'YOUR_AUDIO_TRAJECTORIES_PATH')

audio_dir = os.environ.get("VEX_AUDIO_DIR", 'YOUR_AUDIO_OUTPUT_PATH')

timeout_log = os.environ.get("VEX_TIMEOUT_LOG", 'YOUR_TIMEOUT_LOG_PATH')

# 每个原始 task 最多生成 5 分钟
TIMEOUT_SECONDS = 5 * 60


def main():
    from kokoro import KPipeline
    signal.signal(signal.SIGALRM, timeout_handler)
    pipeline = KPipeline(lang_code='a')
    os.makedirs(audio_dir, exist_ok=True)
    # =========================
    # Load data
    # =========================

    with open(traj_file) as fin:
        data = json.load(fin)


    # 如果之前已经有 timeout 记录，就继续保留
    if os.path.exists(timeout_log):
        with open(timeout_log) as fin:
            timeout_tasks = json.load(fin)
    else:
        timeout_tasks = []


    # 用 set 加速判断，避免重复记录
    timeout_task_ids = {
        item["task"] if isinstance(item, dict) else item
        for item in timeout_tasks
    }


    generated_count = 0
    skipped_count = 0
    timeout_count = 0


    # =========================
    # Main loop
    # =========================

    for datapiece_idx, datapiece in enumerate(tqdm(data)):

        task_id = datapiece["task"]
        start_time = time.time()

        # 给整个 task 设置 5 分钟 alarm
        signal.alarm(TIMEOUT_SECONDS)

        current_outpath = None
        current_step = None

        try:

            for k, step in enumerate(datapiece["steps"]):

                current_step = k

                current_outpath = os.path.join(
                    audio_dir,
                    f"{task_id}_{k}.wav"
                )

                # 无论是否已经存在，都记录路径
                step["audio"] = current_outpath

                # 已存在则跳过
                if os.path.exists(current_outpath):
                    skipped_count += 1
                    continue

                spoken_text = generate(
                    step["description"],
                    pipeline,
                    current_outpath
                )

                if spoken_text != step["description"]:
                    step["audio_text"] = spoken_text
                    print(f"\n[TTS TEXT NORMALIZED] task={task_id}, step={k}")

                generated_count += 1

            # 当前 task 正常完成，取消 alarm
            signal.alarm(0)

        except GenerationTimeout:

            # 一定要取消当前 alarm
            signal.alarm(0)

            timeout_count += 1

            elapsed = time.time() - start_time

            print(
                f"\n[TIMEOUT] task={task_id}, "
                f"step={current_step}, "
                f"elapsed={elapsed:.1f}s"
            )

            # 如果当前 wav 在超时过程中只写了一半，
            # 删除这个可能损坏的文件
            if current_outpath is not None and os.path.exists(current_outpath):
                try:
                    os.remove(current_outpath)
                    print(f"[REMOVE] incomplete file: {current_outpath}")
                except Exception as e:
                    print(
                        f"[WARNING] failed to remove "
                        f"{current_outpath}: {e}"
                    )

            # 记录 timeout task
            if task_id not in timeout_task_ids:

                timeout_info = {
                    "task": task_id,
                    "datapiece_index": datapiece_idx,
                    "step": current_step,
                    "elapsed_seconds": elapsed
                }

                timeout_tasks.append(timeout_info)
                timeout_task_ids.add(task_id)

            # 立刻保存 timeout log，
            # 防止程序之后意外退出导致记录丢失
            with open(timeout_log, "w") as fout:
                json.dump(
                    timeout_tasks,
                    fout,
                    indent=4
                )

            # 跳到下一个 datapiece
            continue

        except Exception as e:

            # 其他异常时也要取消 alarm
            signal.alarm(0)

            print(
                f"\n[ERROR] task={task_id}, "
                f"step={current_step}: {e}"
            )

            raise


    # =========================
    # Save results
    # =========================

    with open(traj_output, "w") as fout:
        json.dump(data, fout, indent=4)


    with open(timeout_log, "w") as fout:
        json.dump(
            timeout_tasks,
            fout,
            indent=4
        )


    print()
    print(f"Generated: {generated_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Timed out tasks: {timeout_count}")
    print(f"Timeout log: {timeout_log}")

if __name__ == "__main__":
    main()
