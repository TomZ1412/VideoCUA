"""Run the mm_agents.qwen CUA harness against a local Qwen3.5 VS checkpoint.

The episode function below follows tests.test_qwen3v_debug; only agent
construction differs. All local weights are loaded before the worker's tasks,
using the same loader as tests.test_vss_local_qwen35.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import shutil
import sys
from datetime import datetime

logger = logging.getLogger(__name__)
CUA_GYM_TASK_ROOT = 'data/cua_gym_tasks'
VIDEO_ROOT = 'data/videos'


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def run_single_example(
        task_config,
        max_steps,
        result_dir,
        args,
        examples,
        trial_id=None,
        local_model=None,
        processor=None,
        video_path=None,
    ):
    # Lazy imports to avoid pulling heavy OSWorld deps during pytest collection.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../OSWorld")))
    # from OSWorld.desktop_env.desktop_env import DesktopEnv
    from desktop_env.desktop_env import DesktopEnv
    from models.qwen35_harness_local import LocalQwen35Agent

    print("task_config", task_config)
    if trial_id is not None:
        print("trial_id", trial_id)
        new_task_name = "{}_sample{}".format(task_config["id"], trial_id)
        result_dir = result_dir / new_task_name
    else:
        result_dir = result_dir / task_config["id"]
    result_dir.mkdir(parents=True, exist_ok=True)

    instruction = task_config["instruction"]
    agent = LocalQwen35Agent(
        local_model=local_model,
        processor=processor,
        model = args.model_name,
        max_tokens = args.max_tokens,
        examples = examples,
        history_n = args.max_history,
        enable_thinking=args.enable_thinking,
        video_path=video_path,
        max_video_pixels=args.max_video_pixels,
        video_max_frames=args.video_max_frames,
        video_min_frames=args.video_min_frames,
        video_interval=args.video_interval,
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



def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', required=True, help='Local Qwen3.5 VS/Audio checkpoint, as in test_vss_local_qwen35')
    parser.add_argument('--lora-ckpt', default='No')
    parser.add_argument('--benchmark', choices=['osworld', 'cuagym'], default='osworld')
    parser.add_argument('--task-config-path', default=None, help='Domain-to-task-ID manifest; default depends on benchmark')
    parser.add_argument('--task-root', default='OSWorld/evaluation_examples/examples')
    parser.add_argument('--cua-gym-task-root', default=CUA_GYM_TASK_ROOT, help='CUA-Gym bundle root: <root>/<task-id>/task.json')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--max-history', '--n-history', dest='max_history', type=int, default=5)
    parser.add_argument('--num-sample', '--nsamples', dest='num_sample', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=65536)
    parser.add_argument('--max-pixels', type=int, default=2076192)
    parser.add_argument('--max-video-pixels', type=int, default=128000)
    parser.add_argument('--use-video', type=str2bool, default=False, help='Include the raw tutorial video in context, as in test_vss_local')
    parser.add_argument('--realvideo', type=str2bool, default=False, help='Use only mapped real tutorials; disable the <video-root>/<task-id>.mp4 fallback')
    parser.add_argument('--video-root', default=VIDEO_ROOT, help='Fallback video directory when --realvideo false')
    parser.add_argument('--video-max-frames', type=int, default=32)
    parser.add_argument('--video-min-frames', type=int, default=2)
    parser.add_argument('--video-interval', type=float, default=2.0, help='Seconds per sampled frame, before the frame cap')
    parser.add_argument('--n-full-res', type=int, default=30)
    parser.add_argument('--enable-thinking', action='store_true', help='Enable local chat-template thinking; disabled by default')
    parser.add_argument('--mode', choices=['debug', 'formal'], default='debug')
    parser.add_argument('--num-workers', type=int, default=1, help='Local model processes; one visible GPU per process')
    parser.add_argument('--worker-index', type=int, default=None, help='Internal or externally launched shard index')
    parser.add_argument('--cache-dir', default='OSWorld/cache')
    parser.add_argument('--result-dir', default='results')
    parser.add_argument('--run-id', default='qwen35_local_qwen_harness')
    parser.add_argument('--oracle-traj', default=None, help='With --use-video true: task-to-video-path JSON, as in test_vss_local; otherwise legacy QwenAgent examples')
    parser.add_argument('--guide-traj', default=None, help='VSS-compatible task-to-description JSON; unmatched tasks run without guidance, ignoring oracle examples (none disables guide mode)')
    return parser


def launch_workers(args, argv):
    """Spawn fresh processes before loading CUDA weights; never fork a model."""
    import torch
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    if len(devices) < args.num_workers:
        raise ValueError(f'{args.num_workers} workers need that many visible GPUs; got {len(devices)}')
    workers = []
    try:
        for index in range(args.num_workers):
            workers.append(subprocess.Popen(
                [sys.executable, '-m', 'tests.test_qwen35_harness_local', *argv, '--worker-index', str(index)],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[index]),
            ))
        return 1 if any([worker.wait() != 0 for worker in workers]) else 0
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            worker.wait()


def resolve_osworld_task(task_root, domain, config_id):
    """Prefer exact configs, falling back to the unverified OSWorld names."""
    domains = dict.fromkeys((domain, domain.removesuffix('_verified')))
    config_ids = dict.fromkeys((config_id, config_id.removesuffix('_verified')))
    candidates = [Path(task_root) / name / f'{task}.json'
                  for name in domains for task in config_ids]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('OSWorld task config not found; tried: ' + ', '.join(map(str, candidates)))


def load_tasks(args):
    tasks = []
    cuagym = args.benchmark == 'cuagym'
    manifest = args.task_config_path or ('cuagym_data/cuagym_data.json' if cuagym else 'OSWorld/evaluation_examples/test_debug.json')
    if cuagym:
        from cua_gym import load_cua_gym_task
        # The adapter replaces its staging directory when loading a task. Keep
        # workers/runs isolated so they cannot remove each other's VM assets.
        import tempfile
        stage_root = tempfile.mkdtemp(prefix='qwen35_cuagym_')
    for domain, task_ids in json.loads(Path(manifest).read_text()).items():
        for task_id in task_ids:
            # Same filename resolution as the API baseline. Preserve verified IDs.
            config_id = task_id.split('_video_')[0].split('_audio_')[0].split('_verified_')[0]
            if cuagym:
                config = load_cua_gym_task(
                    str(Path(args.cua_gym_task_root) / config_id),
                    mock_url_env_file=None, url_variables_manifest=None,
                    use_hosted_defaults=False, ensure_requests=False,
                    cache_root=stage_root,
                )
                # Distinct manifest variants must retain separate resume dirs.
                config['id'] = task_id
            else:
                path = resolve_osworld_task(args.task_root, domain, config_id)
                config = json.loads(path.read_text())
                # Keep variant IDs for guide lookup and separate resume dirs.
                config['id'] = task_id
            tasks.append((domain, config))
    return tasks


def load_guides(path):
    if not path or str(path).lower() == 'none':
        return {}
    guides = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(guides, dict) or any(not isinstance(value, str) for value in guides.values()):
        raise ValueError('--guide-traj must map task IDs to description strings')
    return guides


def task_input_keys(task_id):
    """Try exact video variants and verified aliases before base task aliases."""
    base_id = task_id.split('_video_')[0].split('_audio_')[0]
    plain_id = base_id.removesuffix('_verified')
    suffix = task_id[len(base_id):]
    return tuple(dict.fromkeys((task_id, plain_id + suffix, plain_id + '_verified' + suffix,
                               base_id, plain_id, plain_id + '_verified')))


def task_examples(task_id, examples, guides):
    if guides is not None:
        for key in task_input_keys(task_id):
            if key in guides:
                return {'description': guides[key]}
        logger.info('[No guide] %s: running without guidance', task_id)
        return None
    return next((examples[key] for key in task_input_keys(task_id) if key in examples), None)


def task_video_path(task_id, mapping, args):
    """Resolve exact variants before base IDs, retaining verified aliases."""
    if not args.use_video:
        return None
    base_id = task_id.split('_video_')[0].split('_audio_')[0]
    video = None
    for key in task_input_keys(task_id):
        if key not in mapping:
            continue
        video = mapping[key]
        if isinstance(video, list):
            index = int(task_id.split('_video_')[1]) - 1 if '_video_' in task_id else 0
            video = video[index] if 0 <= index < len(video) else None
        if video is not None and not isinstance(video, str):
            raise ValueError('--oracle-traj must map task IDs to video paths or lists of paths in video mode')
        break
    if video is None and not args.realvideo:
        for key in dict.fromkeys((base_id, base_id.removesuffix('_verified'))):
            candidate = Path(args.video_root) / f'{key}.mp4'
            if candidate.is_file():
                video = str(candidate)
                break
    if video is not None and Path(video).is_file():
        return video
    logger.warning('[No video] %s: %s; running without video context', task_id, video)
    return None


def task_result_dir(result_dir, task_id, trial_id=None):
    name = f'{task_id}_sample{trial_id}' if trial_id is not None else task_id
    return Path(result_dir) / name


def prepare_task_run(result_dir):
    """Skip completed trials; remove partial trials before starting over."""
    result_dir = Path(result_dir)
    if (result_dir / 'result.txt').is_file():
        logger.info('[Skip completed] %s', result_dir)
        return False
    if result_dir.is_symlink():
        result_dir.unlink()
    elif result_dir.exists():
        logger.info('[Restart incomplete] removing %s', result_dir)
        if result_dir.is_dir():
            shutil.rmtree(result_dir)
        else:
            result_dir.unlink()
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.num_workers < 1 or args.num_sample < 1 or args.max_steps < 1 or args.max_tokens < 1 or args.max_history < 0:
        parser.error('Workers, samples, steps and tokens must be positive; history must be nonnegative')
    if not 2 <= args.video_min_frames <= args.video_max_frames or args.video_interval <= 0 or args.max_video_pixels <= 0:
        parser.error('Video frames must satisfy 2 <= min <= max; video interval and pixels must be positive')
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error('worker-index must be in [0, num-workers)')
    if args.mode == 'debug' and args.num_workers != 1:
        parser.error('Use --mode formal for multiple local workers')
    # Validate/read exact task files before loading weights or starting workers.
    tasks = load_tasks(args)
    guides = load_guides(args.guide_traj) if args.guide_traj and args.guide_traj.lower() != 'none' else None
    if args.mode == 'formal' and args.num_workers > 1 and args.worker_index is None:
        return launch_workers(args, argv)
    worker_index = args.worker_index or 0
    tasks = tasks[worker_index::args.num_workers]
    jobs = []
    for domain, config in tasks:
        base_result_dir = Path(args.result_dir) / args.run_id / domain
        for sample in range(args.num_sample):
            trial_id = sample if args.num_sample > 1 else None
            path = task_result_dir(base_result_dir, config['id'], trial_id)
            if (path / 'result.txt').is_file():
                logger.info('[Skip completed] %s', path)
            else:
                jobs.append((config, base_result_dir, trial_id, path))
    if not jobs:
        logger.info('No unfinished tasks for worker %s', worker_index)
        return 0
    oracle = json.loads(Path(args.oracle_traj).read_text()) if (args.use_video or guides is None) and args.oracle_traj and args.oracle_traj.lower() != 'none' else {}
    if not isinstance(oracle, dict):
        parser.error('--oracle-traj must contain a task-keyed JSON object')
    if args.benchmark == 'cuagym':
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'OSWorld'))
        from cua_gym import register_cua_gym_evaluators
        register_cua_gym_evaluators()
    from models.vss_qwen35_runtime import load_model
    from models.qwen35_harness_local import configure_processor
    model, processor, data_args = load_model(
        lora_ckpt=args.lora_ckpt, model_name_or_path=args.model_name,
        max_pixels=args.max_pixels, max_video_pixels=args.max_video_pixels,
        n_full_res=args.n_full_res, use_ttt=False, fixed_memory_size=0,
    )
    configure_processor(processor, data_args)
    # A raw-video mapping is not an image/text demonstration dictionary.
    examples = {} if args.use_video else oracle
    for config, base_result_dir, trial_id, path in jobs:
        video_path = task_video_path(config['id'], oracle, args)
        context = task_examples(config['id'], examples, guides)
        if not args.use_video and isinstance(context, str) and Path(context).suffix.lower() in ('.mp4', '.webm', '.mov', '.mkv', '.avi'):
            context = None
        # Recheck immediately before cleanup in case another run finished it.
        if not prepare_task_run(path):
            continue
        run_single_example(
            config, args.max_steps, base_result_dir,
            args, context, trial_id=trial_id,
            local_model=model, processor=processor,
            video_path=video_path,
        )
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
