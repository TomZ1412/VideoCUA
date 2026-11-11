import sys
import json
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import argparse
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run_single_example(
        task_config,
        max_steps,
        result_dir,
        args,
        examples,
    ):
    # Lazy imports to avoid pulling heavy OSWorld deps during pytest collection.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../OSWorld")))
    # from OSWorld.desktop_env.desktop_env import DesktopEnv
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.gemini import GeminiAgent

    print("task_config", task_config)
    result_dir = result_dir / task_config["id"]
    result_file = result_dir / "result.txt"
    if result_file.exists():
        logger.info("Task id=%s skipped: existing result found at %s", task_config["id"], result_file)
        return
    if result_dir.exists():
        logger.info("Task id=%s has stale outputs without result.txt, removing %s", task_config["id"], result_dir)
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    instruction = task_config["instruction"]
    print("\n\n\n Examples", examples)
    agent = GeminiAgent(
        model = args.model_name,
        max_tokens = args.max_tokens,
        examples = examples,
        history_n = args.max_history,
        temperature = args.temperature,
        api_key = __import__("os").environ.get("OPENAI_API_KEY", ""),
        # base_url = "https://example.com/v1",
        base_url = "https://example.com/v1",
    )
    agent.reset(logger)
    # os.environ.get("VOLCENGINE_IMAGE_ID", "YOUR_IMAGE_ID")
    # image_id = "YOUR_IMAGE_ID"
    # print("Image ID", image_id)
    print(os.environ["PROXY_CONFIG_FILE"])
    print(os.environ['VOLCENGINE_IMAGE_ID'])
    image_id = ""
    env = DesktopEnv(
        path_to_vm=None,
        action_space="pyautogui",
        provider_name="volcengine", 
        region="ap-southeast-3",
        snapshot_name=image_id,
        screen_size=(1920, 1080),
        headless=True,
        os_type="Ubuntu",
        cache_dir=args.cache_dir,
        require_a11y_tree=False,
        enable_proxy=True,
        client_password=os.environ.get("OSWORLD_VM_PASSWORD", ""),
    )
    obs = env.reset(task_config)
    obs = env._get_obs() # Get the initial obs
    action_timestamp = datetime.now().strftime("%Y%m%d@%H%M%S%f")
    with open(os.path.join(result_dir, f"step_0_{action_timestamp}.png"),"wb") as _f:
        _f.write(obs['screenshot'])
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
                # Save screenshot and trajectory information
                with open(os.path.join(result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                            "wb") as _f:
                    _f.write(obs['screenshot'])
                with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({
                        "step_num": step_idx + 1,
                        "action_timestamp": action_timestamp,
                        "action": action,
                        "response": response,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                    }))
                    f.write("\n")

                if done:
                    logger.info("The episode is done.")
                    break

                # Invalid Action
                if not actions:
                    logger.info("Found invalid action!")
                    obs = env._get_obs() # update observation
            if len(actions) > 0:
                step_idx += 1

        if not done: # not completed the task yet
            env.action_history.append('FAIL')

        result = env.evaluate()
        with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"{result}\n")
        logger.info("Result: %.2f", result)
        env.close()
    except Exception as e:
        logger.error(f"failed due to {e}")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single GUI task example")
    parser.add_argument(
        "--task-config-path",
        type=str,
        default="OSWorld/evaluation_examples/test_debug.json",
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
        help="max steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="temperature"
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=5,
        help="max history steps"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=65536,
        help="max steps"
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

    with open(args.task_config_path, "r") as f:
        task_config = json.load(f)
    
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(result_dir / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id
    task_root = Path("OSWorld/evaluation_examples/examples")

    oracle_traj = None
    if os.path.exists(args.oracle_traj):
        with open(args.oracle_traj, "r") as fin:
            oracle_traj = json.load(fin)

    task_to_video = None
    if os.path.exists(args.task_to_video):
        with open(args.task_to_video) as fin:
            task_to_video = json.load(fin)

    # ASR stuff
    asr_replace = None
    if os.path.exists(args.asr_path):
        with open(args.asr_path) as fin:
            asr_replace = json.load(fin)

    if args.mode == "formal":
        # [Formal]
        print(f"[Formal] Eval in parallel with {args.num_workers} workers ...")
        futures = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for key, task_ids in tqdm(task_config.items(), desc="Tasks"):
                # Load configs sequentially (I/O bound), then run tasks concurrently.
                single_task_configs = []
                task_config_to_traj = {}
                for task_id in tqdm(task_ids, desc=f"Task {key}"):
                    with open(task_root / key / f"{task_id.split('_video_')[0].split('_audio_')[0]}.json", "r") as f:
                        single_task_config = json.load(f)
                        single_task_configs.append(single_task_config)

                    # ASR stuff
                    if asr_replace is not None:
                        single_task_config["instruction"] = asr_replace[task_id]
                        if "_audio" in task_id:
                            single_task_config["id"] = task_id.replace("_verified", "")

                    if args.use_video:
                        if task_to_video is not None:
                            video_path = task_to_video[task_id] if task_id in task_to_video else None
                            single_task_config["id"] = task_id.replace("_verified", "")
                        else:
                            video_path = os.path.join('data/videos', single_task_config["id"]+".mp4")
                        if oracle_traj is not None and video_path in oracle_traj:
                            task_config_to_traj[single_task_config["id"]] = oracle_traj[video_path]
                        elif oracle_traj is None and video_path is not None and os.path.exists(video_path):
                            task_config_to_traj[single_task_config["id"]] = video_path
                        else:
                            task_config_to_traj[single_task_config["id"]] = None
                    elif oracle_traj is not None:
                        if single_task_config["id"] in oracle_traj:
                            task_config_to_traj[single_task_config["id"]] = oracle_traj[single_task_config["id"]]
                        else:
                            task_config_to_traj[single_task_config["id"]] = None
                    else:
                        task_config_to_traj[single_task_config["id"]] = None

                key_result_dir = result_dir / run_id / key
                futures_for_task = [
                    executor.submit(
                        run_single_example,
                        single_task_config,
                        result_dir=key_result_dir,
                        max_steps=args.max_steps,
                        args=args,
                        examples=task_config_to_traj[single_task_config["id"]]
                    )
                    for single_task_config in single_task_configs
                ]
                futures.extend(futures_for_task)

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Run"):
                # try:
                fut.result()
                # except Exception as e:
                #     logger.error(f"failed due to {e}")
                #     continue
    elif args.mode == "debug":
        # [DEBUG]
        print(f"[DEBUG] Eval sequentially ...")
        for key, task_ids in tqdm(task_config.items(), desc="Tasks"):
            single_task_configs = []
            task_config_to_traj = {}
            for task_id in tqdm(task_ids, desc=f"Loading configs for {key}"):
                with open(task_root / key / f"{task_id.split('_video_')[0].split('_audio_')[0]}.json", "r") as f:
                    single_task_config = json.load(f)
                    single_task_configs.append(single_task_config)

                # ASR stuff
                if asr_replace is not None:
                    single_task_config["instruction"] = asr_replace[task_id]

                if args.use_video:
                    if task_to_video is not None:
                        video_path = task_to_video[task_id] if task_id in task_to_video else None
                        single_task_config["id"] = task_id.replace("_verified", "")
                    else:
                        video_path = os.path.join('data/videos', single_task_config["id"]+".mp4")
                    if oracle_traj is not None and video_path in oracle_traj:
                        task_config_to_traj[single_task_config["id"]] = oracle_traj[video_path]
                    elif oracle_traj is None and video_path is not None and os.path.exists(video_path):
                        task_config_to_traj[single_task_config["id"]] = video_path
                    else:
                        task_config_to_traj[single_task_config["id"]] = None
                elif oracle_traj is not None:
                    if single_task_config["id"] in oracle_traj:
                        task_config_to_traj[single_task_config["id"]] = oracle_traj[single_task_config["id"]]
                    else:
                        task_config_to_traj[single_task_config["id"]] = None
                else:
                    if "_video_" in task_id:
                        single_task_config["id"] = task_id.replace("_verified", "")
                    task_config_to_traj[single_task_config["id"]] = None
            key_result_dir = result_dir / run_id / key
            for single_task_config in tqdm(single_task_configs, desc=f"Running task {key}"):
                # try:
                run_single_example(
                    task_config=single_task_config,
                    result_dir=key_result_dir,
                    max_steps=args.max_steps,
                    args=args,
                    examples=task_config_to_traj[single_task_config["id"]],
                )
                # except Exception as e:
                #     logging.error(f"Task failed due to: {e}", exc_info=True)
                #     continue
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("All tasks completed.")