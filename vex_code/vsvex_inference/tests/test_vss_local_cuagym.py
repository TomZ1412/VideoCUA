import sys
import json
import os
import random
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import argparse
try:
    import bytedlogger
except ImportError:
    bytedlogger = None
else:
    bytedlogger.config_default()
from tqdm import tqdm
import signal

import torch
from models.vss_local import Qwen3VLAgent as VSSAgent
# from models.qwen3vl_local import Qwen3VLAgent
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwenvl.train.train_qwen import load_model
from qwenvl.data.data_qwen import make_supervised_data_module, LazySupervisedDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from cua_gym import load_cua_gym_task, register_cua_gym_evaluators

CUA_GYM_TASK_ROOT = Path("CUA/data/cua_gym_tasks")


def handler(signum, frame):
    raise TimeoutError("The operation took too long!")


# Register the signal handler
signal.signal(signal.SIGALRM, handler)

logger = logging.getLogger(__name__)


def configure_file_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def resolve_video_path(task_to_video, orig_task_id, base_task_id, video_idx):
    if task_to_video is None:
        return None

    candidates = []
    if orig_task_id in task_to_video:
        candidates.append(task_to_video[orig_task_id])
    if base_task_id in task_to_video:
        candidates.append(task_to_video[base_task_id])

    for value in candidates:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and video_idx is not None:
            try:
                idx = int(video_idx)
                if 0 <= idx < len(value) and isinstance(value[idx], str) and value[idx]:
                    return value[idx]
            except Exception:
                pass

    return None


def run_single_example(
        task_config,
        max_steps,
        result_dir,
        args,
        model,
        processor,
        train_dataset,
        videopath=None,
        video_idx=None,
        sample_idx=0,
        task_guide=None,
        redo_checklist=None,
    ):
    # Lazy imports to avoid pulling heavy OSWorld deps during pytest collection.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../OSWorld")))
    from desktop_env.desktop_env import DesktopEnv

    logger.info("task_config=%s", task_config)
    if video_idx is not None:
        id_str = task_config["id"] + "_video_{}".format(video_idx)
        if sample_idx > 0:
            id_str += "_sample_{}".format(sample_idx)
        result_dir = result_dir / id_str
    else:
        if sample_idx > 0:
            id_str = task_config["id"] + "_sample_{}".format(sample_idx)
        else:
            id_str = task_config["id"]
        result_dir = result_dir / id_str
    result_dir.mkdir(parents=True, exist_ok=True)

    video_path = None
    if args.use_video:
        logger.info("%s", "=" * 90)
        logger.info("videopath=%s", videopath)
        if videopath is not None:
            video_path = videopath
        elif not args.realvideo:
            video_path = os.path.join(
                "OSWorld-self-evolve/data/videos/",
                task_config["id"] + ".mp4"
            )

        if video_path is None or not os.path.exists(video_path):
            logger.info("No Video Found!")
            video_path = None
        else:
            logger.info("Video Path: %s", video_path)
        logger.info("%s", "=" * 90)

    if redo_checklist is not None:
        redo_step = redo_checklist[0]
        redo_action = redo_checklist[1]
    instruction = task_config["instruction"]
    agent = VSSAgent(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        max_tokens=args.max_tokens,
        task_guide=task_guide,
        video_path=video_path,
        history_n=args.n_history,
        compress_history=args.compress_history,
    )
    agent.reset(logger)

    image_id = ""
    env = None
    step_idx = -1
    last_response = None
    last_actions = None
    last_reward = None
    last_done = None
    last_info = None
    try:
        env = DesktopEnv(
            path_to_vm=None,
            action_space="pyautogui",
            provider_name="",
            region="",
            snapshot_name=image_id,
            screen_size=(1920, 1080),
            headless=True,
            os_type="Ubuntu",
            cache_dir=args.cache_dir,
            require_a11y_tree=False,
            enable_proxy=True,
            client_password="",
        )
        obs = env.reset(task_config)
        obs = env._get_obs()
        action_timestamp = datetime.now().strftime("%Y%m%d@%H%M%S%f")
        with open(os.path.join(result_dir, f"step_0_{action_timestamp}.png"), "wb") as _f:
            _f.write(obs['screenshot'])
        obs["screenshot_file"] = os.path.join(result_dir, f"step_0_{action_timestamp}.png")
        done = False
        step_idx = 0
        video_bundle = None
        while not done and step_idx < max_steps:
            action_timestamp = datetime.now().strftime("%Y%m%d@%H%M%S%f")
            # Redo until different
            if redo_checklist is not None and step_idx == redo_step:
                redo_temp = 1.0
                while redo_temp < 1.6:
                    response, actions, video_bundle = agent.predict(
                        instruction,
                        obs,
                        video_bundle=video_bundle,
                        do_sample=True,
                        temperature=redo_temp,
                    )
                    action = response.split("<tool_call>")[1]
                    if action != redo_action:
                        redo_temp += 0.2
                    else:
                        redo_temp += 1.0
            else:
                response, actions, video_bundle = agent.predict(
                    instruction,
                    obs,
                    video_bundle=video_bundle,
                    do_sample=True if sample_idx > 0 else False,
                )
            last_response = response
            last_actions = actions
            logger.info(f"responses: {response}; actions: {actions}")
            for action in actions:
                obs, reward, done, info = env.step(action)
                last_reward = reward
                last_done = done
                last_info = info
                logger.info("Reward: %.2f", reward)
                logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"), "wb") as _f:
                _f.write(obs['screenshot'])
            obs["screenshot_file"] = os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png")
            with open(os.path.join(result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": actions,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }, ensure_ascii=False))
                f.write("\n")

            if done:
                logger.info("The episode is done.")
                break

            # Invalid Action
            if not actions:
                logger.info("Found invalid action!")
                obs = env._get_obs()

            step_idx += 1

        if not done:
            env.action_history.append('FAIL')

        result = env.evaluate()
        with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"{result}\n")
        logger.info("Result: %.2f", result)
    except Exception as e:
        logger.error(
            "run_single_example failed | task_id=%s | sample_idx=%s | video_idx=%s | step_idx=%s | result_dir=%s | video_path=%s | last_reward=%s | last_done=%s | last_response=%r | last_actions=%r | last_info=%r | error=%s",
            task_config.get("id"),
            sample_idx,
            video_idx,
            step_idx,
            result_dir,
            video_path,
            last_reward,
            last_done,
            last_response,
            last_actions,
            last_info,
            e,
            exc_info=True,
        )
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_error:
                logger.error(f"Failed to close environment: {close_error}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single GUI task example")
    parser.add_argument(
        "--task-config-path",
        type=str,
        default="cuagym_data/cuagym_data.json",
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads to run tasks in parallel.",
    )
    parser.add_argument(
        "--n-history",
        type=int,
        default=4,
        help="Number of responses to retain.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="max steps"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="max steps"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="vllm deployed model name",
    )
    parser.add_argument(
        "--lora-ckpt",
        type=str,
        default="No",
        help="vllm deployed model name",
    )
    parser.add_argument(
        "--use-video",
        type=str2bool,
        default=False,
        help="Whether to use video input",
    )
    parser.add_argument(
        "--compress-history",
        type=str2bool,
        default=False,
        help="Whether to compress history, if false, discard history",
    )
    parser.add_argument(
        "--realvideo",
        type=str2bool,
        default=False,
        help="Whether to randomly pick one video input if no direct video found",
    )
    parser.add_argument(
        "--n-full-res",
        type=int,
        default=30,
        help="Past seen screenshot save at lower resolution",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=2076192,
        help="maximum screenshot resolution",
    )
    parser.add_argument(
        "--max-video-pixels",
        type=int,
        default=128000,
        help="maximum screenshot resolution",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        help="mode: debug(sequential) or formal(parallel)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="OSWorld/cache",
        help="cache dir",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="result dir",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="test",
        help="Usually model name (maybe with specific setting)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="none",
        help="Usually model name (maybe with specific setting)",
    )
    parser.add_argument(
        "--oracle-traj",
        type=str,
        default="none",
        help="instructional video paths",
    )
    parser.add_argument(
        "--guide-traj",
        type=str,
        default="none",
        help="text guide from better models",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="guinet",
        help="Which prompt to use",
    )
    parser.add_argument(
        "--sample-checklist",
        type=str,
        default="none",
        help="Sample checklist to evaluate",
    )
    parser.add_argument(
        "--redo-checklist",
        type=str,
        default="none",
        help="Redo task checklist to evaluate",
    )
    parser.add_argument(
        "--use-ttt",
        type=str2bool,
        default=False,
        help="Whether to use TTT prompt",
    )
    parser.add_argument(
        "--memsize",
        type=int,
        default=0,
        help="Memory size for TTT",
    )
    args = parser.parse_args()

    with open(args.task_config_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(result_dir / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_file_logging(output_dir / "run.log")
    run_id = args.run_id
    logger.info("Logging redirected to %s", output_dir / "run.log")

    # Init models
    model, processor, data_args = load_model(
        lora_ckpt=args.lora_ckpt,
        model_name_or_path=args.model_name,
        max_pixels=args.max_pixels,
        max_video_pixels=args.max_video_pixels,
        n_full_res=args.n_full_res,
        use_ttt=args.use_ttt,
        fixed_memory_size=args.memsize
    )
    train_dataset = LazySupervisedDataset(tokenizer=processor, data_args=data_args)

    task_to_guide = None
    task_to_video = None
    if os.path.exists(args.oracle_traj):
        with open(args.oracle_traj, encoding="utf-8") as fin:
            task_to_video = json.load(fin)
    if os.path.exists(args.guide_traj):
        with open(args.guide_traj, encoding="utf-8") as fin:
            task_to_guide = json.load(fin)

    # [DEBUG]
    if args.sample_checklist != "none":
        with open(args.sample_checklist, encoding="utf-8") as fin:
            task_to_checklist = json.load(fin)
    else:
        task_to_checklist = None

    if args.redo_checklist != "none":
        with open(args.redo_checklist, encoding="utf-8") as fin:
            task_to_redo_checklist = json.load(fin)
    else:
        task_to_redo_checklist = None

    register_cua_gym_evaluators()

    logger.info("[DEBUG] Eval sequentially ...")
    for key, task_ids in tqdm(task_config.items(), desc="Tasks"):
        key_result_dir = result_dir / run_id / key
        for task_id in tqdm(task_ids, desc=f"Loading configs for {key}"):
            video_idx = None
            videopath = None
            orig_task_id = task_id
            base_task_id = task_id

            if "_video_" in task_id:
                base_task_id = task_id.split("_video_")[0]
                video_idx = task_id.split("_video_")[1]
                videopath = resolve_video_path(task_to_video, orig_task_id, base_task_id, video_idx)
                task_id = base_task_id

            task_dir = CUA_GYM_TASK_ROOT / task_id
            single_task_config = load_cua_gym_task(
                str(task_dir),
                mock_url_env_file=None,
                url_variables_manifest=None,
                use_hosted_defaults=False,
                ensure_requests=False,
            )

            task_guide = None
            if task_to_guide is not None and task_id in task_to_guide:
                task_guide = task_to_guide[task_id]
            elif task_to_guide is not None and orig_task_id in task_to_guide:
                task_guide = task_to_guide[orig_task_id]

            # Get unfinished sampling tasks
            if task_to_checklist is not None:
                if key in task_to_checklist and task_id in task_to_checklist[key]:
                    sample_ids = task_to_checklist[key][task_id]
                    logger.info("sample_ids=%s", sample_ids)
                else:
                    continue
            else:
                sample_ids = range(args.nsamples)

            # Get redo steps
            redo_checklist = None
            if task_to_redo_checklist is not None and key in task_to_redo_checklist and task_id in task_to_redo_checklist[key]:
                redo_checklist = task_to_redo_checklist[key][task_id]
                logger.info("redo_checklist=%s", redo_checklist)
            elif task_to_redo_checklist is not None:
                continue

            try:
                for sample_idx in sample_ids:
                    run_single_example(
                        task_config=single_task_config,
                        result_dir=key_result_dir,
                        max_steps=args.max_steps,
                        args=args,
                        model=model,
                        processor=processor,
                        train_dataset=train_dataset,
                        videopath=videopath,
                        video_idx=video_idx,
                        sample_idx=sample_idx,
                        task_guide=task_guide,
                        redo_checklist=redo_checklist,
                    )
            except Exception as e:
                logger.error(
                    "Task failed | key=%s | orig_task_id=%s | base_task_id=%s | video_idx=%s | task_dir=%s | videopath=%s | sample_ids=%s | redo_checklist=%s | error=%s",
                    key,
                    orig_task_id,
                    base_task_id,
                    video_idx,
                    task_dir,
                    videopath,
                    list(sample_ids) if not isinstance(sample_ids, range) else f"range({sample_ids.start}, {sample_ids.stop}, {sample_ids.step})",
                    redo_checklist,
                    e,
                    exc_info=True,
                )
                continue

    logger.info("All tasks completed.")
