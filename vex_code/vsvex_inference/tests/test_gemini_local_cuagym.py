import sys
import json
import os
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from cua_gym import load_cua_gym_task, register_cua_gym_evaluators

logger = logging.getLogger(__name__)

CUA_GYM_TASK_ROOT = Path("CUA/data/cua_gym_tasks")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def normalize_task_id(task_id: str) -> str:
    return task_id.replace("_verified", "")


def parse_task_id(task_id: str):
    normalized_task_id = normalize_task_id(task_id)
    base_task_id = normalized_task_id.split("_video_")[0].split("_audio_")[0]
    video_idx = None
    if "_video_" in normalized_task_id:
        video_idx = normalized_task_id.split("_video_")[1].split("_audio_")[0]
    return normalized_task_id, base_task_id, video_idx


def result_already_exists(result_dir: Path, task_id: str) -> bool:
    """Return True if this task already has a completed result.txt."""
    normalized_task_id = normalize_task_id(task_id)
    return (Path(result_dir) / normalized_task_id / "result.txt").is_file()


def resolve_video_path(task_to_video, orig_task_id, normalized_task_id, base_task_id, video_idx):
    if task_to_video is None:
        return None

    candidates = []
    for candidate_key in (orig_task_id, normalized_task_id, base_task_id):
        if candidate_key in task_to_video:
            candidates.append(task_to_video[candidate_key])

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


def load_single_task_config(task_id: str):
    normalized_task_id, base_task_id, video_idx = parse_task_id(task_id)
    task_dir = CUA_GYM_TASK_ROOT / base_task_id
    single_task_config = load_cua_gym_task(
        str(task_dir),
        mock_url_env_file=None,
        url_variables_manifest=None,
        use_hosted_defaults=False,
        ensure_requests=False,
    )
    single_task_config["id"] = normalized_task_id
    return single_task_config, normalized_task_id, base_task_id, video_idx


def maybe_override_instruction(single_task_config, asr_replace, orig_task_id, normalized_task_id, base_task_id):
    if asr_replace is None:
        return

    for candidate_key in (orig_task_id, normalized_task_id, base_task_id):
        if candidate_key in asr_replace:
            single_task_config["instruction"] = asr_replace[candidate_key]
            break


def resolve_examples_for_task(
    single_task_config,
    orig_task_id,
    normalized_task_id,
    base_task_id,
    video_idx,
    args,
    oracle_traj,
    task_to_video,
):
    if args.use_video:
        if task_to_video is not None:
            video_path = resolve_video_path(
                task_to_video,
                orig_task_id,
                normalized_task_id,
                base_task_id,
                video_idx,
            )
        else:
            video_path = os.path.join(
                "OSWorld-self-evolve/data/videos/",
                f"{single_task_config['id']}.mp4",
            )

        if oracle_traj is not None and video_path in oracle_traj:
            return oracle_traj[video_path]
        elif oracle_traj is None and video_path is not None and os.path.exists(video_path):
            return video_path
        else:
            return None

    if oracle_traj is not None:
        for candidate_key in (single_task_config["id"], normalized_task_id, base_task_id):
            if candidate_key in oracle_traj:
                return oracle_traj[candidate_key]

    return None


def run_single_example(
    task_config,
    max_steps,
    result_dir,
    args,
    examples,
):
    # Resume support: skip completed tasks before creating VM / loading heavy deps.
    task_result_dir = Path(result_dir) / task_config["id"]
    result_file = task_result_dir / "result.txt"
    if result_file.is_file():
        logger.info("Skip completed task %s: %s exists", task_config["id"], result_file)
        print(f"[SKIP] {task_config['id']}: {result_file} already exists")
        return

    register_cua_gym_evaluators()

    # Lazy imports to avoid pulling heavy OSWorld deps during pytest collection.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../OSWorld")))
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.gemini import GeminiAgent

    print("task_config", task_config)
    result_dir = task_result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    instruction = task_config["instruction"]
    print("\n\n\n Examples", examples)
    agent = GeminiAgent(
        model=args.model_name,
        max_tokens=args.max_tokens,
        examples=examples,
        history_n=args.max_history,
        temperature=args.temperature,
        api_key=os.environ.get("GEMINI_API_KEY"),
        base_url="base_url",
    )
    agent.reset(logger)

    image_id = ""
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
        _f.write(obs["screenshot"])

    done = False
    step_idx = 0
    try:
        while not done and step_idx < max_steps:
            action_timestamp = datetime.now().strftime("%Y%m%d@%H%M%S%f")
            response, actions = agent.predict(
                instruction,
                obs,
            )
            logger.info(f"responses: {response}; actions: {actions}")
            for action in actions:
                obs, reward, done, info = env.step(action)
                logger.info("Reward: %.2f", reward)
                logger.info("Done: %s", done)

                with open(
                    os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                    "wb",
                ) as _f:
                    _f.write(obs["screenshot"])

                with open(os.path.join(result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step_num": step_idx + 1,
                                "action_timestamp": action_timestamp,
                                "action": action,
                                "response": response,
                                "reward": reward,
                                "done": done,
                                "info": info,
                                "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                            },
                            ensure_ascii=False,
                        )
                    )
                    f.write("\n")

                if done:
                    logger.info("The episode is done.")
                    break

                if not actions:
                    logger.info("Found invalid action!")
                    obs = env._get_obs()

            if len(actions) > 0:
                step_idx += 1

        if not done:
            env.action_history.append("FAIL")

        result = env.evaluate()
        with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"{result}\n")
        logger.info("Result: %.2f", result)
        env.close()
    except Exception as e:
        logger.error(f"failed due to {e}", exc_info=True)
        env.close()


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
        default=4,
        help="Number of worker threads to run tasks in parallel.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="max steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="temperature",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=5,
        help="max history steps",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=65536,
        help="max steps",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-3.5-flash",
        help="vllm deployed model name",
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
        "--use-video",
        type=str2bool,
        default=False,
        help="Whether to use video input",
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
        help="Usually model name (maybe with specific setting)",
    )
    parser.add_argument(
        "--asr-path",
        type=str,
        default="none",
        help="Usually model name (maybe with specific setting)",
    )
    parser.add_argument(
        "--task-to-video",
        type=str,
        default="none",
        help="Path to the task to video mapping file",
    )
    args = parser.parse_args()

    with open(args.task_config_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(result_dir / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id

    oracle_traj = None
    if os.path.exists(args.oracle_traj):
        with open(args.oracle_traj, "r", encoding="utf-8") as fin:
            oracle_traj = json.load(fin)

    task_to_video = None
    if os.path.exists(args.task_to_video):
        with open(args.task_to_video, "r", encoding="utf-8") as fin:
            task_to_video = json.load(fin)

    asr_replace = None
    if os.path.exists(args.asr_path):
        with open(args.asr_path, "r", encoding="utf-8") as fin:
            asr_replace = json.load(fin)

    register_cua_gym_evaluators()

    if args.mode == "formal":
        print(f"[Formal] Eval in parallel with {args.num_workers} workers ...")
        futures = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for key, task_ids in tqdm(task_config.items(), desc="Tasks"):
                single_task_configs = []
                task_config_to_traj = {}
                key_result_dir = result_dir / run_id / key

                for task_id in tqdm(task_ids, desc=f"Task {key}"):
                    if result_already_exists(key_result_dir, task_id):
                        print(
                            f"[SKIP] {normalize_task_id(task_id)}: "
                            f"{key_result_dir / normalize_task_id(task_id) / 'result.txt'} already exists"
                        )
                        continue

                    orig_task_id = task_id
                    single_task_config, normalized_task_id, base_task_id, video_idx = load_single_task_config(task_id)
                    maybe_override_instruction(
                        single_task_config,
                        asr_replace,
                        orig_task_id,
                        normalized_task_id,
                        base_task_id,
                    )
                    single_task_configs.append(single_task_config)

                    task_config_to_traj[single_task_config["id"]] = resolve_examples_for_task(
                        single_task_config=single_task_config,
                        orig_task_id=orig_task_id,
                        normalized_task_id=normalized_task_id,
                        base_task_id=base_task_id,
                        video_idx=video_idx,
                        args=args,
                        oracle_traj=oracle_traj,
                        task_to_video=task_to_video,
                    )

                futures_for_task = [
                    executor.submit(
                        run_single_example,
                        single_task_config,
                        result_dir=key_result_dir,
                        max_steps=args.max_steps,
                        args=args,
                        examples=task_config_to_traj[single_task_config["id"]],
                    )
                    for single_task_config in single_task_configs
                ]
                futures.extend(futures_for_task)

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Run"):
                fut.result()

    elif args.mode == "debug":
        print(f"[DEBUG] Eval sequentially ...")
        for key, task_ids in tqdm(task_config.items(), desc="Tasks"):
            single_task_configs = []
            task_config_to_traj = {}
            key_result_dir = result_dir / run_id / key

            for task_id in tqdm(task_ids, desc=f"Loading configs for {key}"):
                if result_already_exists(key_result_dir, task_id):
                    print(
                        f"[SKIP] {normalize_task_id(task_id)}: "
                        f"{key_result_dir / normalize_task_id(task_id) / 'result.txt'} already exists"
                    )
                    continue

                orig_task_id = task_id
                single_task_config, normalized_task_id, base_task_id, video_idx = load_single_task_config(task_id)
                maybe_override_instruction(
                    single_task_config,
                    asr_replace,
                    orig_task_id,
                    normalized_task_id,
                    base_task_id,
                )
                single_task_configs.append(single_task_config)

                task_config_to_traj[single_task_config["id"]] = resolve_examples_for_task(
                    single_task_config=single_task_config,
                    orig_task_id=orig_task_id,
                    normalized_task_id=normalized_task_id,
                    base_task_id=base_task_id,
                    video_idx=video_idx,
                    args=args,
                    oracle_traj=oracle_traj,
                    task_to_video=task_to_video,
                )

            for single_task_config in tqdm(single_task_configs, desc=f"Running task {key}"):
                run_single_example(
                    task_config=single_task_config,
                    result_dir=key_result_dir,
                    max_steps=args.max_steps,
                    args=args,
                    examples=task_config_to_traj[single_task_config["id"]],
                )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("All tasks completed.")
