"""Launch vLLM's own OpenAI API server with audio-visual preprocessing."""
import argparse
import json
import os
import sys
from pathlib import Path
from .data import add_media_args, media_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tp', '--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--served-model-name', default='qwen35-audio')
    parser.add_argument('--max-model-len', type=int, default=155000)
    parser.add_argument('--max-num-seqs', type=int, default=None)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=-1)
    parser.add_argument('--allowed-local-media-path', default=None)
    parser.add_argument('--mm-processor-cache-gb', type=float, default=8)
    parser.add_argument('--mm-processor-cache-type', choices=('lru', 'shm'), default='shm')
    parser.add_argument('--mm-shm-cache-max-object-size-mb', type=int, default=2047)
    parser.add_argument('--enforce-eager', action='store_true')
    add_media_args(parser)
    args, extra = parser.parse_known_args()
    from . import ensure_plugin_installed
    ensure_plugin_installed()
    os.environ['QWENVL_MODEL'] = str(Path(args.model).resolve())
    os.environ['QWENVL_MEDIA_OPTIONS'] = json.dumps(media_options(args))
    os.environ['QWENVL_USE_AUDIO'] = '0' if args.no_audio else '1'
    os.environ['VLLM_VIDEO_LOADER_BACKEND'] = 'qwen35_audio'
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    from . import register
    register()
    command = ['vllm', 'serve', args.model]
    command.extend(['--tensor-parallel-size', str(args.tp)])
    for key in ('host', 'port', 'served_model_name', 'max_model_len', 'max_num_seqs', 'gpu_memory_utilization'):
        if getattr(args, key) is not None:
            command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
    command.extend(['--media-io-kwargs', '{"video": {"video_backend": "qwen35_audio"}}',
                    '--limit-mm-per-prompt', '{"video": 1}',
                    '--mm-processor-cache-gb', str(args.mm_processor_cache_gb),
                    '--mm-processor-cache-type', args.mm_processor_cache_type,
                    '--generation-config', 'vllm',
                    '--override-generation-config', json.dumps(dict(
                        max_new_tokens=args.max_tokens, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k)),
                    '--default-chat-template-kwargs', '{"enable_thinking": false}'])
    if args.mm_processor_cache_type == 'shm':
        command.extend(['--mm-shm-cache-max-object-size-mb', str(args.mm_shm_cache_max_object_size_mb)])
    if args.allowed_local_media_path:
        command.extend(['--allowed-local-media-path', args.allowed_local_media_path])
    if args.enforce_eager:
        command.append('--enforce-eager')
    command.extend(extra[1:] if extra and extra[0] == '--' else extra)
    sys.argv = command
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == '__main__':
    main()
