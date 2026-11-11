#!/usr/bin/env python3
"""Shard verifier input and run one independent model replica per GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import read_records


def split_records(records, count):
    # Balanced contiguous slices, without duplicating the final example.
    quotient, remainder = divmod(len(records), count)
    start = 0
    shards = []
    for rank in range(count):
        end = start + quotient + (rank < remainder)
        shards.append((start, records[start:end]))
        start = end
    return shards


def materialize(path, kind):
    if not path or not path.startswith('hdfs://'):
        return path
    destination = ROOT / ('dataset' if kind == 'data' else 'output') / Path(path.rstrip('/')).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        subprocess.run(['hdfs', 'dfs', 'get', path, str(destination)], check=True)
    return str(destination)


def merge_results(output, total_ranks):
    output = Path(output)
    with (output.parent / '.merge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = [output.parent / f'test_results_rank{rank}.json' for rank in range(total_ranks)]
        missing = [p.name for p in files if not p.is_file()]
        if missing:
            print(f'Waiting for other nodes: missing {", ".join(missing)}', flush=True)
            return False
        rows = []
        for path in files:
            rows.extend(read_records(path))
        if any(row.get('error') for row in rows):
            raise ValueError('Rank results contain errors; refusing to publish a complete merged result')
        rows.sort(key=lambda row: row['index'])
        indices = [row['index'] for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError('Duplicate input indices across ranks')
        if output.exists():
            return True  # Another node already merged under this lock.
        with output.open('x') as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        print(f'Merged {len(rows)} results into {output}', flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_base', '--processor', dest='processor')
    parser.add_argument('--lora_ckpt')
    parser.add_argument('--dataset', '--input', required=True)
    parser.add_argument('--run_name', default='qwen35_verifier')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--num_gpus', '--gpus-per-node', type=int,
                        default=int(os.environ.get('ARNOLD_WORKER_GPU', '8')))
    parser.add_argument('--nnodes', type=int, default=int(os.environ.get('ARNOLD_WORKER_NUM', '1')))
    parser.add_argument('--node_rank', type=int, default=int(os.environ.get('ARNOLD_ID', '0')))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--model_type', choices=('dense', 'moe'), help='Legacy compatibility; checked against model config')
    parser.add_argument('--total_timestamps', type=int, help='Legacy option; native Qwen3.5 uses per-patch timestamps')
    args, worker_args = parser.parse_known_args()
    if (args.num_gpus < 1 or args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes
            or args.start < 0 or args.limit < 0):
        parser.error('Invalid GPU/node count, node rank or slice')
    if '--help' in worker_args:
        return
    output = args.output or ROOT / 'output/test' / args.run_name / 'test_results.json'
    if output.exists():
        parser.error(f'Output already exists: {output}; use a new --run_name or --output')
    if args.total_timestamps is not None:
        print('--total_timestamps is unused: Qwen3.5 uses native temporal-patch timestamps.', flush=True)
    model = materialize(args.model, 'model')
    processor = materialize(args.processor, 'model')
    lora = materialize(args.lora_ckpt, 'model')
    dataset_path = materialize(args.dataset, 'data')
    if args.model_type:
        config = json.loads((Path(model) / 'config.json').read_text())
        expected = {'qwen3_5': 'dense', 'qwen3_5_moe': 'moe'}.get(config.get('model_type'))
        if args.model_type != expected:
            parser.error('--model_type disagrees with checkpoint config')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(args.num_gpus)]
    if len(devices) < args.num_gpus:
        parser.error(f'{args.num_gpus} GPUs requested but CUDA_VISIBLE_DEVICES={visible}')
    records = read_records(dataset_path)
    end = min(len(records), args.start + args.limit) if args.limit else len(records)
    selected = records[args.start:end]
    if not selected:
        parser.error('Input slice is empty')
    total_ranks = args.num_gpus * args.nnodes
    shards = split_records(selected, total_ranks)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / 'dataset'
    shard_dir.mkdir(exist_ok=True)
    workers = []

    def stop_workers(signum=None, frame=None):
        for process, _, log in workers:
            if process.poll() is None:
                process.terminate()
        for process, _, log in workers:
            process.wait()
            log.close()
        if signum is not None:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_workers)
    signal.signal(signal.SIGINT, stop_workers)
    try:
        for local_rank in range(args.num_gpus):
            rank = args.node_rank * args.num_gpus + local_rank
            offset, shard = shards[rank]
            rank_output = output.parent / f'test_results_rank{rank}.json'
            if rank_output.exists() or rank_output.with_suffix('.json.jsonl').exists():
                raise ValueError(f'Rank {rank} output already exists; use a new run name')
            shard_path = shard_dir / f'{rank}.json'
            with shard_path.open('x') as handle:
                json.dump(shard, handle, ensure_ascii=False)
            if not shard:
                rank_output.write_text('[]\n')
                continue
            command = [sys.executable, str(ROOT / 'verify.py'), '--model', model,
                       '--dataset', str(shard_path), '--output', str(rank_output),
                       '--index-offset', str(args.start + offset), *worker_args]
            if processor:
                command += ['--model_base', processor]
            if lora:
                command += ['--lora_ckpt', lora]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': devices[local_rank], 'PYTHONUNBUFFERED': '1'}
            log_path = output.parent / f'rank{rank}.log'
            log = log_path.open('x')
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            workers.append((process, rank, log))
            print(f'Rank {rank}: GPU={devices[local_rank]}, examples={len(shard)}, log={log_path}', flush=True)
        failures = []
        for process, rank, log in workers:
            status = process.wait()
            log.close()
            if status:
                failures.append((rank, status))
        if failures:
            raise RuntimeError(f'Workers failed: {failures}; see rank*.log. Merged output was not published.')
        merge_results(output, total_ranks)
    finally:
        stop_workers()


if __name__ == '__main__':
    main()
