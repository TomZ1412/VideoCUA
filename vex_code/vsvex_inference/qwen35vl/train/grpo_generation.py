"""Colocated vLLM generation following TRL's external-launcher/TP protocol.

Reference: huggingface/trl, trl/generation/vllm_generation.py (main).
Weights are materialized one at a time, including from CPU-offloaded DTensors.
"""
import gc
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from qwenvl.train.grpo_transport import broadcast_cpu_tensor, broadcast_object


class GRPOGeneration:
    def __init__(self, model_path, args, rank, local_size):
        from qwenvl.vllm_audio import register
        from vllm import LLM

        register()
        tp = args.vllm_tensor_parallel_size
        if tp < 1 or local_size % tp:
            raise ValueError('vllm_tensor_parallel_size must divide the GPUs per node')
        self.tp = tp
        timeout = timedelta(seconds=args.ddp_timeout)
        self.sync_group = dist.new_group(backend='nccl', timeout=timeout)
        self.tp_group = None
        # Every training rank creates groups in the same order.
        for start in range(0, dist.get_world_size(), tp):
            group = dist.new_group(list(range(start, start + tp)), backend='nccl', timeout=timeout)
            if start <= rank < start + tp:
                self.tp_group, self.tp_root = group, start
        self.llm = LLM(
            model=model_path, tensor_parallel_size=tp,
            distributed_executor_backend='external_launcher',
            seed=args.seed + rank // tp,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            max_num_seqs=256,
            enable_sleep_mode=True, enforce_eager=args.vllm_enforce_eager,
            # vLLM V1 fans out n completions into separate requests. Reuse
            # their shared prompt blocks, as in the evaluation server.
            enable_prefix_caching=True,
            limit_mm_per_prompt={'video': 1}, mm_processor_cache_gb=0,
            logprobs_mode='processed_logprobs',
        )
        self.llm.sleep(level=2)
        self.args = args
        # Persistent across sync_vllm calls; every TP peer advances together.
        self._dummy_round = 0

    @staticmethod
    def full(tensor):
        if tensor.is_cpu:
            # A blocking copy also waits for earlier work on this stream,
            # including the previous parameter's asynchronous NCCL gather.
            tensor = tensor.to(torch.device('cuda', torch.cuda.current_device()), non_blocking=True)
        return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

    def _weight_sync_barrier(self):
        dist.barrier(group=self.sync_group, device_ids=[torch.cuda.current_device()])

    @torch.no_grad()
    def sync_weights(self, model, active=True):
        self.active = active
        self._kv_awake = False
        self._weight_sync_barrier()
        gc.collect()
        torch.cuda.empty_cache()
        if active:
            self.llm.wake_up(tags=['weights'])
        torch.cuda.synchronize()
        self._weight_sync_barrier()
        target = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        # DTensor gathers are world collectives, including ranks with no prompts.
        state = model.state_dict()
        modules = {name.replace('._checkpoint_wrapped_module', ''): module
                   for name, module in model.named_modules()}
        for raw_name, value in state.items():
            if 'lora_' in raw_name or 'original_module' in raw_name:
                continue
            name = raw_name.replace('._checkpoint_wrapped_module', '')
            name = name.removeprefix('base_model.model.').replace('.base_layer', '')
            name = name.replace('.modules_to_save.default', '')
            value = self.full(value)
            if raw_name.endswith('.base_layer.weight'):
                layer = modules[raw_name.replace('._checkpoint_wrapped_module', '').removesuffix('.base_layer.weight')]
                if getattr(layer, 'use_dora', {}).get('default', False):
                    raise ValueError('GRPO weight synchronization does not support DoRA')
                a = self.full(layer.lora_A['default'].weight)
                b = self.full(layer.lora_B['default'].weight)
                delta = b.float() @ a.float()
                if layer.fan_in_fan_out:
                    delta = delta.T
                value = value + (delta * layer.scaling['default']).to(value.dtype)
            if active:
                target.load_weights([(name, value)])
        if active:
            # Cached KV/SSM states belong to the previous policy weights.
            self.llm.reset_prefix_cache()
        del state
        # Finish device work on every rank before any replica wakes KV memory
        # and switches from FSDP collectives to vLLM generation.
        torch.cuda.synchronize()
        self._weight_sync_barrier()
        torch.cuda.empty_cache()

    def _broadcast_items(self, items):
        # This scheduler assigns prompts to the TP leader. Broadcast tensor
        # payloads directly: all_gather_object pads *every* rank to the largest
        # serialized video, multiplying multi-GB input buffers by TP size.
        root = dist.get_rank() == self.tp_root
        keys = ('input_ids', 'pixel_values_videos', 'video_grid_thw',
                'pixel_values', 'image_grid_thw', 'audio_feature')
        metadata = [(index, item['ref'], {key: (tuple(item[key].shape), item[key].dtype)
                                         for key in keys if item.get(key) is not None})
                     for index, item in items] if root else None
        metadata = broadcast_object(metadata, src=self.tp_root, group=self.tp_group)
        result = []
        for offset, (index, reference, specs) in enumerate(metadata):
            item = {'ref': reference}
            for key, (shape, dtype) in specs.items():
                tensor = items[offset][1][key].cpu().contiguous() if root else torch.empty(shape, dtype=dtype)
                broadcast_cpu_tensor(tensor, src=self.tp_root, group=self.tp_group)
                item[key] = tensor
            result.append((index, item))
        return result

    def generate(self, items):
        from vllm import SamplingParams
        from qwenvl.vllm_audio.data import vllm_prompt

        # Match TRL: each TP participant submits exactly the same ordered prompts.
        started = time.perf_counter()
        items = self._broadcast_items(items)
        if dist.get_rank() == self.tp_root:
            print(f'[GRPO TP root={self.tp_root}] prompt broadcast={time.perf_counter() - started:.3f}s, '
                  f'prompts={len(items)}', flush=True)
        if not items:
            return []
        params = SamplingParams(
            n=self.args.num_generations, temperature=self.args.temperature,
            top_p=self.args.top_p, top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
            max_tokens=self.args.max_output_token, logprobs=0,
        )
        self.llm.wake_up(tags=['kv_cache'])
        self._kv_awake = True
        started = time.perf_counter()
        outputs = self.llm.generate([vllm_prompt(item) for _, item in items], params, use_tqdm=False)
        if dist.get_rank() == self.tp_root:
            print(f'[GRPO TP root={self.tp_root}] real generation={time.perf_counter() - started:.3f}s', flush=True)
        packages = []
        if dist.get_rank() == self.tp_root:
            for (index, item), output in zip(items, outputs, strict=True):
                completions = []
                for completion in output.outputs:
                    logps = [p[t].logprob for t, p in zip(completion.token_ids, completion.logprobs, strict=True)]
                    completions.append(dict(ids=list(completion.token_ids), logps=logps, text=completion.text))
                packages.append(dict(index=index, prompt_length=item['input_ids'].shape[1],
                                     reference=item['ref'], completions=completions))
        return packages

    def sleep(self):
        if self.active:
            self.llm.sleep(level=2)
        gc.collect()
        torch.cuda.empty_cache()

    def sync_vllm(self):
        """Keep finished TP replicas prefilling until all real generation ends."""
        if not self.args.sync_vllm:
            return
        from vllm import SamplingParams

        # Idle replicas have current weights too, but have not awakened KV yet.
        if not self._kv_awake:
            self.llm.wake_up(tags=['kv_cache'])
            self._kv_awake = True
        length = min(24000, self.args.vllm_max_model_len - 1)
        prompt = {'prompt_token_ids': ([123, 456, 789] * ((length + 2) // 3))[:length]}
        params = SamplingParams(temperature=1.0, max_tokens=1, top_k=1, top_p=1.0,
                                stop_token_ids=[], n=1,
                                repetition_penalty=self.args.repetition_penalty)
        device = torch.cuda.current_device()
        done = torch.zeros(1, dtype=torch.int32, device=torch.device('cuda', device)) if self.tp > 1 else None
        # Enqueue the world wait separately so dummy forwards can keep using
        # the current stream. Join it only after all replicas have arrived.
        sync_stream = torch.cuda.Stream(device=device)
        sync_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sync_stream):
            handle = dist.barrier(group=self.sync_group, device_ids=[device], async_op=True)
        rounds = 0
        while True:
            # Completion can become visible at different times on TP peers.
            # Only the leader chooses whether the *whole group* calls generate.
            if self.tp == 1:
                finished = handle.is_completed()
            else:
                if dist.get_rank() == self.tp_root:
                    done.fill_(int(handle.is_completed()))
                dist.broadcast(done, src=self.tp_root, group=self.tp_group)
                finished = bool(done.item())
            if finished:
                break
            # Salt is hashed into the first block and thus all descendants.
            # A new salt guarantees full prefill without disabling the real
            # rollout cache or drawing from training/sampling RNG streams.
            prompt['cache_salt'] = f'grpo-sync-{self.tp_root}-{self._dummy_round}'
            self._dummy_round += 1
            # Never return these outputs as training rollouts.
            self.llm.generate([prompt], sampling_params=params, use_tqdm=False)
            rounds += 1
        handle.wait()
        if dist.get_rank() == self.tp_root:
            print(f'[sync_vllm TP root={self.tp_root}] dummy prefill rounds={rounds}', flush=True)
