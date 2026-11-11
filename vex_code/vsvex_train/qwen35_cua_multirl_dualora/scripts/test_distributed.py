"""Arnold multi-node evaluation: slice JSON locally, merge once/node, serve replicas, upload results."""
import argparse
from collections import deque
import concurrent.futures
import contextlib
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HDFS_ROOT = os.environ.get('HDFS_ROOT', '')


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', '--dataset', '--data', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--model_base', '--model-base')
    p.add_argument('--lora_ckpt', '--lora-ckpt')
    p.add_argument('--merge_dir', '--merge-dir')
    p.add_argument('--run_name', '--run-name', required=True)
    p.add_argument('--output_dir', '--output-dir', default=str(ROOT / 'output/test'))
    p.add_argument('--cache_dir', '--cache-dir', default=str(ROOT / 'output/test/.cache'))
    p.add_argument('--nnodes', '--num_machines', type=int, default=int(os.getenv('ARNOLD_WORKER_NUM', '1')))
    p.add_argument('--node_rank', '--node-rank', type=int, default=int(os.getenv('ARNOLD_ID', '0')))
    p.add_argument('--tp_size', '--tp-size', '--tp', type=int, default=1)
    p.add_argument('--gpus_per_node', '--gpus-per-node', type=int, default=8)
    p.add_argument('--base_port', '--base-port', type=int, default=8000)
    p.add_argument('--concurrency', type=int, default=4, help='Concurrent requests per model replica')
    p.add_argument('--max_num_seqs', '--max-num-seqs', type=int, default=None,
                   help='Maximum sequences per replica; defaults to concurrency')
    p.add_argument('--max_model_len', '--max-model-len', type=int, default=120000)
    p.add_argument('--max_num_batched_tokens', '--max-num-batched-tokens', type=int, default=16384)
    p.add_argument('--gpu_memory_utilization', '--gpu-memory-utilization', type=float, default=0.9)
    p.add_argument('--mm_processor_cache_gb', '--mm-processor-cache-gb', type=float, default=8)
    p.add_argument('--mm_processor_cache_type', '--mm-processor-cache-type', choices=('lru', 'shm'), default='shm')
    p.add_argument('--mm_shm_cache_max_object_size_mb', '--mm-shm-cache-max-object-size-mb', type=int, default=2047)
    p.add_argument('--enforce_eager', '--enforce-eager', action='store_true')
    p.add_argument('--max_tokens', '--max-tokens', type=int, default=40960)
    p.add_argument('--temperature', type=float)
    p.add_argument('--top_p', '--top-p', type=float)
    p.add_argument('--top_k', '--top-k', type=int, default=20)
    p.add_argument('--presence_penalty', '--presence-penalty', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=0)
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument('--think', nargs='?', const=True, default=False, type=boolean)
    thinking.add_argument('--no_think', '--no-think', dest='think', action='store_false')
    sampling = p.add_mutually_exclusive_group()
    sampling.add_argument('--interval', type=float, default=0.1)
    sampling.add_argument('--fps', type=float, help='Equivalent to interval=1/fps')
    p.add_argument('--max_frames', '--max-frame', '--max-frames', type=int, default=1024)
    p.add_argument('--min_frames', '--min-frame', '--min-frames', type=int, default=4)
    p.add_argument('--max_pixels', '--max-pixel', '--max-pixels', type=int, default=176400)
    p.add_argument('--min_pixels', '--min-pixel', '--min-pixels', type=int, default=4096)
    p.add_argument('--no_audio', '--no-audio', action='store_true')
    p.add_argument('--startup_timeout', '--startup-timeout', type=float, default=1800)
    p.add_argument('--request_timeout', '--request-timeout', type=float, default=1800)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--hdfs_root', '--hdfs-root', default=HDFS_ROOT)
    p.add_argument('--no_hdfs_upload', '--no-hdfs-upload', action='store_true')
    p.add_argument('--dry_run', '--dry-run', action='store_true', help='Slice local input and write plans; do not load GPUs, merge, or upload')
    args = p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        p.error('run_name must start with a letter/digit and contain only letters, digits, _, -, .')
    if not 0 <= args.node_rank < args.nnodes:
        p.error('Require 0 <= node_rank < nnodes')
    if args.tp_size <= 0 or args.gpus_per_node <= 0 or args.gpus_per_node % args.tp_size:
        p.error('gpus_per_node must be a positive multiple of tp_size')
    if args.fps is not None:
        if args.fps <= 0:
            p.error('fps must be positive')
        args.interval = 1 / args.fps
    if args.interval <= 0 or not 1 <= args.min_frames <= args.max_frames:
        p.error('Require interval > 0 and 1 <= min_frames <= max_frames')
    if not 1024 <= args.min_pixels <= args.max_pixels:
        p.error('Require 1024 <= min_pixels <= max_pixels')
    if min(args.concurrency, args.max_tokens, args.retries, args.max_num_batched_tokens) <= 0:
        p.error('Concurrency, token budgets, and retries must be positive')
    if args.max_num_seqs is None:
        args.max_num_seqs = args.concurrency
    if args.max_num_seqs <= 0:
        p.error('max_num_seqs must be positive')
    if not 0 <= args.mm_processor_cache_gb < float('inf') or args.mm_shm_cache_max_object_size_mb <= 0:
        p.error('Multimodal cache size must be finite/nonnegative; maximum object size must be positive')
    if args.max_tokens >= args.max_model_len:
        p.error('max_model_len must also leave room for the input prompt')
    if min(args.startup_timeout, args.request_timeout) <= 0 or not 0 < args.gpu_memory_utilization < 1:
        p.error('Timeouts must be positive; GPU memory utilization must be between 0 and 1')
    if not 1024 <= args.base_port <= 65535 - args.gpus_per_node // args.tp_size + 1:
        p.error('Invalid replica port range')
    if args.temperature is None:
        args.temperature = 1.0 if args.think else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if args.think else 0.8
    if not args.no_hdfs_upload and not args.hdfs_root.startswith('hdfs://'):
        p.error('hdfs_root must be an HDFS URI')
    return args


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def materialize(source, cache_dir):
    """Download each immutable HDFS URI once; publish only a completed copy."""
    if not source.startswith('hdfs://'):
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    from cruise.utilities.hdfs_io import hcopy
    parent = Path(cache_dir).resolve() / digest(source)[:24]
    target = parent / source.rstrip('/').rsplit('/', 1)[-1]
    with file_lock(parent / '.download.lock'):
        if target.exists() and (parent / '.complete').exists():
            return target
        with tempfile.TemporaryDirectory(dir=parent, prefix='.download-') as stage:
            if not hcopy(source, stage):
                raise RuntimeError(f'HDFS download failed: {source}')
            downloaded = Path(stage) / target.name
            if not downloaded.exists():
                raise RuntimeError(f'HDFS copy did not create {downloaded}')
            if target.exists():
                raise RuntimeError(f'Incomplete cache at {target}; remove it before retrying')
            downloaded.rename(target)
        (parent / '.complete').write_text(source + '\n')
    return target


def load_shard(args):
    """Each node independently reads the same JSON and selects its balanced slice."""
    source = materialize(args.input, args.cache_dir)
    samples = json.loads(source.read_text())
    if not isinstance(samples, list) or not all(isinstance(row, dict) for row in samples):
        raise ValueError('Input JSON must be an array of sample objects')
    quotient, remainder = divmod(len(samples), args.nnodes)
    start = args.node_rank * quotient + min(args.node_rank, remainder)
    size = quotient + (args.node_rank < remainder)
    return {
        'start': start,
        'items': samples[start:start + size],
        'source_dir': None if args.input.startswith('hdfs://') else str(source.parent),
    }


def gpu_groups(args):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = [d.strip() for d in visible.split(',')] if visible else [str(i) for i in range(args.gpus_per_node)]
    if len(devices) < args.gpus_per_node or any(not d.strip() or d == '-1' for d in devices):
        raise ValueError('CUDA_VISIBLE_DEVICES exposes fewer GPUs than gpus_per_node')
    devices = devices[:args.gpus_per_node]
    if len(set(devices)) != len(devices):
        raise ValueError('CUDA_VISIBLE_DEVICES contains duplicate devices')
    return [devices[i:i + args.tp_size] for i in range(0, len(devices), args.tp_size)]


def checkpoint_stamp(path):
    return [(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.jinja')]


def prepare_model(args, rank_dir):
    model = materialize(args.model, args.cache_dir)
    model_base = materialize(args.model_base, args.cache_dir) if args.model_base else None
    adapter = materialize(args.lora_ckpt, args.cache_dir) if args.lora_ckpt else None
    if (model / 'adapter_config.json').is_file():
        if adapter is not None and adapter != model:
            raise ValueError('model is already an adapter; do not supply a different lora_ckpt')
        adapter = model
        saved_base = json.loads((adapter / 'adapter_config.json').read_text())['base_model_name_or_path']
        model = model_base or materialize(saved_base, args.cache_dir)
    if adapter is None:
        return model
    request = dict(model=str(model), model_base=str(model_base) if model_base else None,
                   adapter=str(adapter), stamps=[checkpoint_stamp(p) for p in dict.fromkeys([model, adapter] + ([model_base] if model_base else []))])
    signature = digest(request)
    output = Path(args.merge_dir).resolve() if args.merge_dir else rank_dir / 'merged_model'
    with file_lock(output.with_name(output.name + '.lock')):
        marker = output / 'test_merge.json'
        if marker.exists():
            if json.loads(marker.read_text()).get('signature') != signature:
                raise ValueError(f'Merged model source changed: choose a different merge_dir ({output})')
            return output
        if output.exists() and any(output.iterdir()):
            raise ValueError(f'Unverified existing merge directory: {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix='.merge-') as stage:
            command = [sys.executable, str(ROOT / 'scripts/merge_base_model.py'),
                       '--model_name_or_path', str(model), '--lora_ckpt', str(adapter), '--output_dir', stage]
            if model_base:
                command.extend(['--model_base', str(model_base)])
            subprocess.run(command, cwd=ROOT, stderr=subprocess.STDOUT, check=True)
            write_json(Path(stage) / 'test_merge.json', dict(signature=signature, **request))
            if output.exists():
                output.rmdir()  # Only the empty directory accepted above.
            Path(stage).rename(output)
    return output


def server_command(args, model, port):
    command = [sys.executable, '-m', 'qwenvl.vllm_audio.serve', '--model', str(model),
               '--tp', str(args.tp_size), '--host', '127.0.0.1', '--port', str(port),
               '--served-model-name', 'qwen35-test', '--allowed-local-media-path', '/',
               '--interval', str(args.interval), '--min-frame', str(args.min_frames),
               '--max-frame', str(args.max_frames), '--min-pixel', str(args.min_pixels),
               '--max-pixel', str(args.max_pixels), '--max-model-len', str(args.max_model_len),
               '--max-tokens', str(args.max_tokens),
               '--gpu-memory-utilization', str(args.gpu_memory_utilization),
               '--max-num-batched-tokens', str(args.max_num_batched_tokens),
               '--mm-processor-cache-gb', str(args.mm_processor_cache_gb),
               '--mm-processor-cache-type', args.mm_processor_cache_type,
               '--mm-shm-cache-max-object-size-mb', str(args.mm_shm_cache_max_object_size_mb),
               '--reasoning-parser', 'qwen3', '--enable-prefix-caching']
    if args.max_num_seqs is not None:
        command.extend(['--max-num-seqs', str(args.max_num_seqs)])
    if args.no_audio:
        command.append('--no-audio')
    if args.enforce_eager:
        command.append('--enforce-eager')
    return command


class Replicas:
    def __init__(self, args, model, rank_dir):
        self.args, self.model, self.rank_dir = args, model, rank_dir
        self.processes, self.endpoints = [], []
        self.closed = False

    def __enter__(self):
        try:
            from qwenvl.vllm_audio import ensure_plugin_installed
            ensure_plugin_installed()
            groups = gpu_groups(self.args)
            # Check all ports before starting any GPU workers.
            for i in range(len(groups)):
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', self.args.base_port + i))
            for i, devices in enumerate(groups):
                port = self.args.base_port + i
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices),
                           VLLM_WORKER_MULTIPROC_METHOD='spawn')
                for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                    env.pop(key, None)
                process = subprocess.Popen(server_command(self.args, self.model, port), cwd=ROOT, env=env,
                                           stdout=sys.stderr, stderr=subprocess.STDOUT, start_new_session=True)
                self.processes.append(process)
                self.endpoints.append(f'http://127.0.0.1:{port}/v1')
                print(f'rank {self.args.node_rank}: replica {i}, GPUs={devices}, port={port}, PID={process.pid}', flush=True)
            pending = set(range(len(groups)))
            deadline = time.monotonic() + self.args.startup_timeout
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while pending:
                self.check()
                for i in list(pending):
                    try:
                        with opener.open(self.endpoints[i].removesuffix('/v1') + '/health', timeout=2) as response:
                            if response.status == 200:
                                pending.remove(i)
                    except (OSError, urllib.error.URLError):
                        pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Replicas {sorted(pending)} not ready; inspect stdout logs')
                if pending:
                    time.sleep(2)
            return self
        except BaseException:
            self.close()
            raise

    def check(self):
        for i, process in enumerate(self.processes):
            if process.poll() is not None:
                raise RuntimeError(f'Replica {i} exited ({process.returncode}); see stdout logs')

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Terminate entire process groups, including vLLM engine/TP children.
        for process in self.processes:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 20
        for process in self.processes:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
        for process in self.processes:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def __exit__(self, *exc):
        self.close()


def extract_sample(sample):
    conversations = sample.get('conversations', [])
    prompt = next((turn for turn in conversations if turn.get('from', turn.get('role')) in ('human', 'user')), None)
    if prompt is None:
        raise ValueError('Sample has no human prompt in conversations')
    ref = next((turn.get('value', turn.get('content')) for turn in conversations
                if turn.get('from', turn.get('role')) in ('gpt', 'assistant')), None)
    return dict(tos_key=sample.get('tos_key'), video=sample.get('video'), prompt=prompt, ref=ref)


@contextlib.contextmanager
def video_url(sample, args, rank_dir, source_dir):
    video = sample.get('video')
    if video:
        if not isinstance(video, str):
            raise ValueError('Each sample must contain one video path')
        if video.startswith(('http://', 'https://', 'file://', 'data:')):
            yield video
        else:
            if not video.startswith('hdfs://') and not Path(video).is_absolute():
                if not source_dir:
                    raise ValueError('Relative video paths require a local input JSON; use absolute paths with HDFS input')
                video = str(Path(source_dir) / video)
            yield materialize(video, args.cache_dir).as_uri()
    elif sample.get('tos_key'):
        from qwenvl.data.data_qwen import get_tos_client
        response = get_tos_client().get_object(sample['tos_key'])
        with tempfile.NamedTemporaryFile(suffix='.mp4', dir=rank_dir) as handle:
            handle.write(response.data)
            handle.flush()
            yield Path(handle.name).as_uri()
    else:
        raise ValueError('Sample requires video or tos_key')


def request_kwargs(sample, args, url):
    result = extract_sample(sample)
    text = result['prompt'].get('value', result['prompt'].get('content'))
    text = text.replace('<image>', '').replace('<video>', '').strip()
    return dict(model='qwen35-test', messages=[{'role': 'user', 'content': [
        {'type': 'video_url', 'video_url': {'url': url}}, {'type': 'text', 'text': text}]}],
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
        presence_penalty=args.presence_penalty, seed=args.seed,
        extra_body={'top_k': args.top_k, 'chat_template_kwargs': {'enable_thinking': args.think},
                    'media_io_kwargs': {'video': {'use_audio': not args.no_audio and sample.get('use_audio', True)}}})


class NoFinalAnswer(ValueError):
    pass


def response_parts(message):
    """Keep the server's reasoning separate from the final answer."""
    content = message.content or ''
    reasoning = (getattr(message, 'reasoning_content', None)
                 or getattr(message, 'reasoning', None) or '')
    if '</think>' in content:
        prefix, content = content.split('</think>', 1)
        reasoning = reasoning or prefix.removeprefix('<think>').strip()
    elif '<think>' in content:
        reasoning = reasoning or content.split('<think>', 1)[1].strip()
        content = ''
    return content.strip(), reasoning


def evaluation_signature(args, shard, routing='video_groups_work_stealing_v2'):
    keys = ('model', 'model_base', 'lora_ckpt', 'tp_size', 'think', 'max_tokens', 'temperature',
            'top_p', 'top_k', 'presence_penalty', 'seed', 'interval', 'min_frames', 'max_frames',
            'min_pixels', 'max_pixels', 'no_audio', 'max_model_len', 'max_num_batched_tokens',
            'enforce_eager', 'nnodes', 'node_rank', 'max_num_seqs', 'concurrency', 'gpus_per_node')
    return digest(dict(options={k: getattr(args, k) for k in keys}, shard=shard,
                       prefix_caching=True, routing=routing))


def read_completed(journal, expected_indices):
    completed = {}
    if journal.exists():
        lines = journal.read_text().splitlines()
        for number, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if number == len(lines) - 1:
                    break  # A final partial write from a terminated process.
                raise
            if record['index'] not in expected_indices:
                raise ValueError('Journal contains an index outside this shard')
            if not record.get('error'):
                completed[record['index']] = record
    return completed


def video_group_key(sample, source_dir, index):
    """Identify shared videos without opening or downloading the media."""
    video = sample.get('video')
    if isinstance(video, str) and video:
        if video.startswith('file://'):
            from urllib.parse import unquote, urlparse
            parsed = urlparse(video)
            if parsed.netloc not in ('', 'localhost'):
                return ('video', video)
            video = unquote(parsed.path)
        if '://' not in video and not video.startswith('data:'):
            path = Path(video)
            if not path.is_absolute() and source_dir:
                path = Path(source_dir) / path
            video = str(path.resolve())
        return ('video', video)
    if not video and sample.get('tos_key'):
        return ('tos', sample['tos_key'])
    return ('invalid', index)


def assign_video_groups(shard, replica_count, completed=()):
    """Initially balance pending video groups by count, preserving video affinity."""
    groups = {}
    for offset, sample in enumerate(shard['items']):
        index = shard['start'] + offset
        if index in completed:
            continue
        key = video_group_key(sample, shard['source_dir'], index)
        groups.setdefault(key, []).append((index, sample))
    assignments = [[] for _ in range(replica_count)]
    for group in sorted(groups.values(), key=len, reverse=True):
        target = min(range(replica_count), key=lambda i: len(assignments[i]))
        assignments[target].extend(group)
    return assignments


@dataclass(eq=False)
class PendingVideoGroup:
    items: deque
    started: bool = False


class VideoWorkScheduler:
    """Prefer local videos; let idle request slots steal only unsent work.

    Whole unstarted groups move first to retain cache locality. If only started
    groups remain, split a pending tail so a large video cannot strand replicas.
    Acquiring/removing work is atomic; HTTP requests never hold the lock.
    """

    def __init__(self, shard, replica_count, completed=()):
        self.lock = threading.Lock()
        self.queues = []
        self.pending = []
        self.active = [0] * replica_count
        self.finished = [0] * replica_count
        self.stolen = [0] * replica_count
        for assigned in assign_video_groups(shard, replica_count, completed):
            groups = {}
            for index, sample in assigned:
                key = video_group_key(sample, shard['source_dir'], index)
                groups.setdefault(key, deque()).append((index, sample))
            self.queues.append(deque(PendingVideoGroup(items) for items in groups.values()))
            self.pending.append(len(assigned))

    def take(self, replica):
        with self.lock:
            local = self.queues[replica]
            if not local:
                donor = max(range(len(self.queues)), key=self.pending.__getitem__)
                if not self.pending[donor]:
                    return None  # All remaining work is already in flight.
                remote = self.queues[donor]
                group = next((g for g in reversed(remote) if not g.started), None)
                if group is not None:
                    remote.remove(group)
                else:
                    source = max(remote, key=lambda g: len(g.items))
                    tail = deque()
                    for _ in range((len(source.items) + 1) // 2):
                        tail.appendleft(source.items.pop())
                    if not source.items:
                        remote.remove(source)
                    group = PendingVideoGroup(tail)
                moved = len(group.items)
                local.append(group)
                self.pending[donor] -= moved
                self.pending[replica] += moved
                self.stolen[replica] += moved
            group = local[0]
            group.started = True
            item = group.items.popleft()
            if not group.items:
                local.popleft()
            self.pending[replica] -= 1
            self.active[replica] += 1
            return item

    def finish(self, replica):
        with self.lock:
            self.active[replica] -= 1
            self.finished[replica] += 1

    def status(self):
        with self.lock:
            return ' | '.join(
                f'replica {i}: pending={self.pending[i]}, active={self.active[i]}, '
                f'done={self.finished[i]}, stolen={self.stolen[i]}'
                for i in range(len(self.queues)))


def evaluate(args, shard, rank_dir, replicas, completed):
    from openai import OpenAI
    import httpx
    records = dict(completed)
    journal = rank_dir / 'progress.jsonl'
    lock = threading.Lock()
    cancelled = threading.Event()
    scheduler = VideoWorkScheduler(shard, len(replicas.endpoints), completed)
    print(f'rank {args.node_rank}: scheduler=video_groups_work_stealing_v2; {scheduler.status()}', flush=True)
    total = len(shard['items'])

    def worker(replica, endpoint, handle):
        with OpenAI(base_url=endpoint, api_key='YOUR_API_KEY', max_retries=0, timeout=args.request_timeout,
                    http_client=httpx.Client(trust_env=False, timeout=args.request_timeout)) as client:
            while not cancelled.is_set():
                item = scheduler.take(replica)
                if item is None:
                    return
                index, sample = item
                started = time.monotonic()
                error, result, metadata = None, None, {}
                try:
                    result = extract_sample(sample)
                    for attempt in range(args.retries):
                        try:
                            replicas.check()
                            with video_url(sample, args, rank_dir, shard['source_dir']) as url:
                                response = client.chat.completions.create(**request_kwargs(sample, args, url))
                            choice = response.choices[0]
                            result['pred'], result['reasoning'] = response_parts(choice.message)
                            metadata = dict(finish_reason=choice.finish_reason,
                                            usage=response.usage.model_dump() if response.usage else None)
                            if not result['pred']:
                                hint = ('Increase max_tokens for thinking.' if choice.finish_reason == 'length' else
                                        'The checkpoint stopped before emitting a final-answer section.')
                                raise NoFinalAnswer(f'No final answer (finish_reason={choice.finish_reason}). {hint}')
                            break
                        except Exception as exc:
                            if (cancelled.is_set() or isinstance(exc, NoFinalAnswer)
                                    or getattr(exc, 'status_code', None) in (400, 401, 403, 404, 422)
                                    or attempt + 1 == args.retries):
                                raise
                            time.sleep(min(5 * (attempt + 1), 15))
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    result = result or dict(tos_key=sample.get('tos_key'), video=sample.get('video'), prompt=None, ref=None)
                    result.update(pred='', error=error)
                    result.setdefault('reasoning', '')
                record = dict(index=index, replica=replica, result=result, error=error, **metadata,
                              seconds=time.monotonic() - started)
                with lock:
                    records[index] = record
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    handle.flush()
                    if len(records) % 10 == 0 or len(records) == total or error:
                        print(f"rank {args.node_rank}: {len(records)}/{total}, replica={replica}, "
                              f"index={index}, seconds={record['seconds']:.1f}, error={error}", flush=True)
                scheduler.finish(replica)

    # Rewriting the validated records also removes an interrupted trailing line.
    with journal.open('w') as handle:
        for index in sorted(completed):
            handle.write(json.dumps(completed[index], ensure_ascii=False) + '\n')
        handle.flush()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(replicas.endpoints) * args.concurrency) as executor:
            futures = [executor.submit(worker, i, endpoint, handle)
                       for i, endpoint in enumerate(replicas.endpoints)
                       for _ in range(args.concurrency)]
            try:
                pending = set(futures)
                last_status = time.monotonic()
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    replicas.check()
                    if time.monotonic() - last_status >= 30 or not pending:
                        print(f'rank {args.node_rank}: {scheduler.status()}', flush=True)
                        last_status = time.monotonic()
            except BaseException:
                cancelled.set()
                replicas.close()
                raise
    return records


def upload_result(path, args):
    from cruise.utilities.hdfs_io import hcopy, hmkdir, hexists
    remote_dir = args.hdfs_root.rstrip('/') + '/' + args.run_name
    remote = remote_dir + '/' + path.name
    if not hmkdir(remote_dir):
        raise RuntimeError(f'Cannot create {remote_dir}')
    if not hcopy(str(path), remote_dir) or not hexists(remote):
        raise RuntimeError(f'HDFS upload failed: {remote}; local result preserved at {path}')
    print(f'Uploaded {remote}', flush=True)
    return remote


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.output_dir).resolve() / args.run_name
    rank_dir = run_dir / f'rank{args.node_rank}'
    rank_dir.mkdir(parents=True, exist_ok=True)
    # Catch duplicate launch on the same machine/shared rank directory.
    with (rank_dir / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shard = load_shard(args)
        write_json(rank_dir / 'input.json', shard['items'])
        plan = dict(vars(args), samples=len(shard['items']), start_index=shard['start'],
                    gpu_groups=gpu_groups(args), replicas=args.gpus_per_node // args.tp_size,
                    hdfs_result=f'{args.hdfs_root.rstrip("/")}/{args.run_name}/test_result_rank{args.node_rank}.json')
        write_json(rank_dir / 'plan.json', plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            return
        signature = evaluation_signature(args, shard)
        manifest = rank_dir / 'evaluation.json'
        journal = rank_dir / 'progress.jsonl'
        if args.resume and manifest.exists():
            saved_signature = json.loads(manifest.read_text())['signature']
            # Allow old static-routing runs to resume with the same input and
            # generation settings; servers restart, so their caches are cold.
            if saved_signature not in (signature, evaluation_signature(args, shard, routing='video_groups_v1')):
                raise ValueError('Resume parameters/input changed; choose a new run_name')
        elif args.resume and journal.exists():
            raise ValueError('Cannot resume a journal without its evaluation manifest')
        elif manifest.exists() or journal.exists():
            raise ValueError('Existing run: use --resume or a new run_name')
        write_json(manifest, dict(signature=signature, plan=plan))
        indices = set(range(shard['start'], shard['start'] + len(shard['items'])))
        records = read_completed(journal, indices) if args.resume else {}
        if indices - records.keys():
            model = prepare_model(args, rank_dir)
            with Replicas(args, model, rank_dir) as replicas:
                records = evaluate(args, shard, rank_dir, replicas, records)
        if set(records) != indices:
            raise RuntimeError('Missing or duplicated result indices')
        output = run_dir / f'test_result_rank{args.node_rank}.json'
        write_json(output, [records[i]['result'] for i in sorted(records)])
        errors = sum(bool(record.get('error')) for record in records.values())
        summary = dict(rank=args.node_rank, samples=len(records), errors=errors, thinking=args.think,
                       output=str(output), hdfs_result=None)
        write_json(rank_dir / 'summary.json', summary)
        if not args.no_hdfs_upload:
            summary['hdfs_result'] = upload_result(output, args)
            write_json(rank_dir / 'summary.json', summary)
        print(f'Finished rank {args.node_rank}: {len(records)} samples, {errors} errors -> {output}', flush=True)
        if summary['hdfs_result']:
            print(f"HDFS result directory: {summary['hdfs_result'].rsplit('/', 1)[0]}/", flush=True)
        if errors:
            raise RuntimeError(f'{errors} samples failed; results include errors. Use --resume to retry.')


if __name__ == '__main__':
    # Python context managers then clean up all child process groups on SIGTERM.
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    main()
