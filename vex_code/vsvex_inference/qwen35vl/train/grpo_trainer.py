"""Single-turn audio/video GRPO with node-local rollout redistribution."""
import logging
import math
import os
import time
from itertools import groupby

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from qwenvl.data.grpo_data import collate_prompt_indices
from qwenvl.data.grpo_prefetch import MediaWorker, prefetch_media
from qwenvl.train.grpo_generation import GRPOGeneration
from qwenvl.train.grpo_batching import batch_inputs, response_batches
from qwenvl.train.grpo_utils import NodePromptSampler, balance_rollout_groups, dynamic_group_valid, group_advantages, rewards
from qwenvl.train.trainer import FiniteLossTrainer
from qwenvl.train.grpo_transport import broadcast_object, exchange_rollout_groups, transfer_object


class GRPOTrainer(FiniteLossTrainer):
    def __init__(self, *args, **kwargs):
        generation_model_path = kwargs.pop('generation_model_path')
        config = kwargs['args']
        self.rollout_batch_size = config.per_device_train_batch_size
        self.rollout_accumulation = config.gradient_accumulation_steps
        self.local_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
        if config.grpo_response_batch_size < 1:
            raise ValueError('grpo_response_batch_size must be positive')
        if config.num_generations < 2 or config.max_output_token < 1 or config.temperature <= 0:
            raise ValueError('GRPO requires num_generations >= 2, max_output_token > 0, temperature > 0')
        if config.importance_sampling_level not in ('token', 'sequence'):
            raise ValueError('importance_sampling_level must be token or sequence')
        if config.loss_type not in ('dapo', 'grpo'):
            raise ValueError('loss_type must be dapo or grpo')
        if not math.isfinite(config.temperature) or not 0 < config.vllm_gpu_memory_utilization <= 1:
            raise ValueError('temperature must be finite and GPU memory utilization must be in (0, 1]')
        if config.vllm_max_model_len <= config.max_output_token:
            raise ValueError('vllm_max_model_len must leave room for both prompt and completion')
        if not math.isfinite(config.vllm_importance_sampling_cap) or config.vllm_importance_sampling_cap <= 0:
            raise ValueError('vllm_importance_sampling_cap must be finite and positive')
        self.reward_weights = [float(x) for x in config.reward_weights.split(',')]
        if len(self.reward_weights) != 3 or not all(math.isfinite(x) for x in self.reward_weights):
            raise ValueError('reward_weights must contain finite length,format,accuracy weights')
        # Trainer owns optimizer/scheduler/checkpoints; all rollout accumulation
        # happens inside a single training_step, with a global count agreement.
        config.per_device_train_batch_size = 1
        config.gradient_accumulation_steps = 1
        super().__init__(*args, **kwargs)
        if not self.is_fsdp_enabled or self.accelerator.state.fsdp_plugin.fsdp_version != 2:
            raise ValueError('GRPO requires FSDP2')
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.node_group = None
        for start in range(0, dist.get_world_size(), self.local_size):
            group = dist.new_group(list(range(start, start + self.local_size)), backend='nccl')
            if start <= dist.get_rank() < start + self.local_size:
                self.node_group = group
        self.prompt_sampler = NodePromptSampler(
            self.train_dataset.cached_token_lengths, dist.get_rank(), dist.get_world_size(),
            self.local_size, self.rollout_batch_size, self.rollout_accumulation,
            config.num_generations, seed=config.data_seed if config.data_seed is not None else config.seed,
            window_batches=config.length_grouping_window, modalities=self.train_dataset.modalities,
            oversample_factor=config.dynamic_sampling_oversample_factor if config.dynamic_sampling else 1.0,
            initial_accept_rate=config.dynamic_sampling_initial_accept_rate if config.dynamic_sampling else None,
            max_oversample_factor=config.dynamic_sampling_max_factor,
        )
        self.generation = GRPOGeneration(generation_model_path, config, dist.get_rank(), self.local_size)
        from qwenvl.train.grpo_loss import GRPOLoss
        self.grpo_loss = GRPOLoss(temperature=config.temperature, epsilon=config.epsilon,
                                  importance_cap=config.vllm_importance_sampling_cap,
                                  importance_sampling_level=config.importance_sampling_level,
                                  loss_type=config.loss_type)
        self.model_accepts_loss_kwargs = False
        self.prompt_sampler.plan_mode = True
        self.media_worker = None

    def train(self, *args, **kwargs):
        self.media_worker = MediaWorker(self.train_dataset)
        try:
            return super().train(*args, **kwargs)
        finally:
            self.media_worker.close()

    def _assigned_indices(self, indices):
        rank_in_node = dist.get_rank() % self.local_size
        tp = self.args.vllm_tensor_parallel_size
        if rank_in_node % tp:
            return []
        engine, engines = rank_in_node // tp, self.local_size // tp
        return [index for offset, index in enumerate(indices) if offset % engines == engine]

    def _prefetch_next(self, next_key):
        if next_key is None or self.state.global_step + 1 >= self.state.max_steps:
            self.media_worker.plan([])
            return
        indices = self.prompt_sampler.resolve_plan(next_key)
        assigned = self._assigned_indices(indices)
        self.media_worker.plan(assigned)
        if dist.get_rank() % self.args.vllm_tensor_parallel_size == 0:
            print(f'[GRPO rank={dist.get_rank()}] next-step media plan={len(assigned)} prompts, '
                  f'node candidates={len(indices)}, key={next_key}', flush=True)

    def get_train_dataloader(self):
        # Each rank receives the same node plan. Wrapping this loader in
        # Accelerate would incorrectly shard that plan a second time. Media is
        # loaded by the separate eight-thread prefetch_media pipeline below;
        # spawning DataLoader workers here would only prefetch integer indices.
        loader = DataLoader(self.train_dataset, sampler=self.prompt_sampler,
                            batch_size=1, collate_fn=collate_prompt_indices, num_workers=0)
        loader.set_epoch = self.prompt_sampler.set_epoch
        return loader

    def _rollouts(self, indices, next_plan_key=None):
        rank_in_node = dist.get_rank() % self.local_size
        tp = self.args.vllm_tensor_parallel_size
        engine = rank_in_node // tp
        items = []
        started = time.perf_counter()
        if rank_in_node % tp == 0:
            assigned = self._assigned_indices(indices)
            self.media_worker.plan(assigned)
            entries = ((index, None) for index in assigned)
            with prefetch_media(self.train_dataset, entries, worker=self.media_worker) as prepared:
                items = [(entry[0], item) for entry, item in prepared]
            print(f'[GRPO rank={dist.get_rank()}] media preparation={time.perf_counter() - started:.3f}s', flush=True)
        if self.is_world_process_zero():
            print(f"[GRPO step {self.state.global_step}] Generating {len(indices)} prompts/node x "
                  f"{self.args.num_generations} completions; TP={tp}, "
                  f"CPU offload={self.args.grpo_offload_model}", flush=True)
        if self.args.grpo_offload_model:
            self.model.to('cpu')
        try:
            self.generation.sync_weights(self.model, active=engine < len(indices) or self.args.sync_vllm)
            # Forecast the next step with the last known acceptance while this
            # step generates. Only CPU IO runs on the worker; no collectives.
            self._prefetch_next(next_plan_key)
            groups = self.generation.generate(items)
            items.clear()  # Release generation media before training redistribution.
            self.generation.sync_vllm()
        finally:
            self.generation.sleep()
            if self.args.grpo_offload_model:
                self.model.to(self.accelerator.device)
        node_start = dist.get_rank() - rank_in_node
        started = time.perf_counter()
        # Score/filter at the generating TP leader. Only small scheduling
        # metadata is replicated; completion IDs/logprobs go to one owner.
        selected, processed, fallback_meta = [], [], None
        reward_sums = [0.0, 0.0, 0.0]
        generated = 0
        completion_tokens = prompt_tokens = 0
        for offset, group in enumerate(groups):
            if len(group['completions']) != self.args.num_generations:
                raise RuntimeError('vLLM returned an incomplete generation group')
            components = [rewards(c['text'], group['reference'], len(c['ids']), self.args.max_output_token)
                          for c in group['completions']]
            values = [sum(w * r for w, r in zip(self.reward_weights, row)) for row in components]
            advantages, _ = group_advantages(values)
            valid_group = dynamic_group_valid([row[2] for row in components])
            generated += len(values)
            completion_tokens += sum(len(c['ids']) for c in group['completions'])
            prompt_tokens += group['prompt_length'] * len(values)
            for row in components:
                reward_sums = [a + b for a, b in zip(reward_sums, row)]
            packages = []
            for completion, advantage in zip(group['completions'], advantages, strict=True):
                if not completion['ids']:
                    raise RuntimeError('vLLM returned an empty completion')
                packages.append(dict(index=group['index'], ids=completion['ids'], logps=completion['logps'],
                                     advantage=advantage, length=group['prompt_length'] + len(completion['ids'])))
            processed.append(packages)
            meta = dict(index=group['index'], source=rank_in_node, offset=offset,
                        prompt_length=group['prompt_length'], count=len(packages))
            if fallback_meta is None or meta['prompt_length'] < fallback_meta['prompt_length']:
                fallback_meta = meta
            if not self.args.dynamic_sampling or valid_group:
                selected.append(meta)
        metadata, fallbacks = [], []
        for src in range(node_start, node_start + self.local_size, tp):
            summary = broadcast_object((selected, fallback_meta) if dist.get_rank() == src else None,
                                       src, self.node_group)
            metadata.extend(summary[0])
            if summary[1] is not None:
                fallbacks.append(summary[1])
        if not fallbacks:
            raise RuntimeError('Node generated no rollouts')
        assignments = balance_rollout_groups(metadata, self.local_size)
        local_groups = exchange_rollout_groups(processed, metadata, assignments, self.node_group, node_start)
        local = [package for group in local_groups for package in group]
        batches = response_batches(local_groups, self.args.grpo_response_batch_size)
        # Idle ranks must still execute the same modality branches under FSDP.
        # Send just one zero-weight completion to them; active ranks reuse their
        # last owned video and never load somebody else's video for padding.
        fallback = local[-1] if local else None
        shortest = min(fallbacks, key=lambda g: g['prompt_length'])
        for dst, assigned in enumerate(assignments):
            if assigned:
                continue
            src = shortest['source']
            payload = processed[shortest['offset']][0] if rank_in_node == src else None
            received = transfer_object(payload, node_start + src, node_start + dst, self.node_group)
            if rank_in_node == dst:
                fallback = received
        del groups, processed, local_groups
        if rank_in_node == 0:
            print(f'[GRPO node root={node_start}] rollout exchange={time.perf_counter() - started:.3f}s', flush=True)
        stats = torch.tensor([*reward_sums, generated, completion_tokens, prompt_tokens],
                             device=self.accelerator.device, dtype=torch.float64)
        dist.all_reduce(stats)
        self._rollout_metrics = {f'rewards/{name}': stats[i].item() / max(stats[3].item(), 1)
                                 for i, name in enumerate(('length', 'format', 'accuracy'))}
        self._rollout_metrics.update(completion_tokens=stats[4].item() / max(stats[3].item(), 1),
                                     prompt_tokens=stats[5].item() / max(stats[3].item(), 1))
        # Agree before *any* training forward, including zero-acceptance nodes.
        counts = torch.tensor([len(local), sum(len(p['ids']) for p in local), len(batches)],
                              device=self.accelerator.device, dtype=torch.long)
        all_counts = [torch.zeros_like(counts) for _ in range(dist.get_world_size())]
        dist.all_gather(all_counts, counts)
        total_tokens = sum(int(x[1].item()) for x in all_counts)
        iterations = max(1, max(int(x[2].item()) for x in all_counts))
        counts = [int(x[0].item()) for x in all_counts]
        total = sum(counts)
        self._rollout_metrics.update(accepted_rollouts=total, accepted_completion_tokens=total_tokens,
                                     forward_backward_per_rank=iterations,
                                     acceptance_rate=total / max(stats[3].item(), 1))
        if self.args.dynamic_sampling:
            self.prompt_sampler.update_acceptance(self._rollout_metrics['acceptance_rate'])
            self._rollout_metrics.update(
                sampled_prompts_per_node=len(indices),
                target_prompts_per_node=self.prompt_sampler.target_prompts_per_node,
                next_prompts_per_node=self.prompt_sampler.prompts_per_node,
            )
        # Refine the same stable plan with the just-measured acceptance. New
        # supplemental reads overlap this step's forward/backward.
        self._prefetch_next(next_plan_key)
        batches += [[dict(fallback, advantage=0.0, padding=True)] for _ in range(iterations - len(batches))]
        logging.info('GRPO step=%d rank=%d accepted=%d forward_backward=%d global_rollouts=%d',
                     self.state.global_step, dist.get_rank(), counts[dist.get_rank()], iterations, total)
        if self.is_world_process_zero():
            print(f"[GRPO step {self.state.global_step}] Rollouts ready: accepted={total}/{int(stats[3].item())}, "
                  f"forward/backward per rank={iterations}, "
                  f"mean completion tokens={self._rollout_metrics['completion_tokens']:.1f}", flush=True)
        return batches, total, total_tokens

    def log(self, logs, *args, **kwargs):
        if 'loss' in logs:
            logs.update(getattr(self, '_rollout_metrics', {}))
        return super().log(logs, *args, **kwargs)

    def _loss(self, model, packages, item):
        inputs = batch_inputs(packages, item, shared_prefix=self.args.grpo_response_batch_size > 1)
        inputs = self._prepare_inputs(inputs)
        return model(**inputs, grpo_loss=self.grpo_loss)

    def training_step(self, model, inputs, num_items_in_batch=None):
        indices = self.prompt_sampler.resolve_plan(inputs['plan_key'])
        batches, total, total_tokens = self._rollouts(indices, inputs['next_plan_key'])
        model.train()
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
        result = torch.zeros((), device=self.accelerator.device)
        entries = ((index, list(group)) for index, group in groupby(batches, key=lambda batch: batch[0]['index']))
        with prefetch_media(self.train_dataset, entries, worker=self.media_worker) as prepared:
            for (_, group), item in prepared:
                for batch in group:
                    with self.compute_loss_context_manager():
                        loss = self._loss(model, batch, item)
                        # FSDP averages ranks. DAPO weights each local token mean
                        # by its length / global accepted tokens; GRPO averages
                        # completion means. Padding contributes to neither sum.
                        if self.args.loss_type == 'dapo':
                            token_count = sum(len(p['ids']) for p in batch if not p.get('padding', False))
                            loss = loss * (dist.get_world_size() * token_count / max(total_tokens, 1))
                        else:
                            # Padding batches contain one zero-advantage response.
                            loss = loss * dist.get_world_size() * len(batch) / max(total, 1)
                    if not torch.isfinite(loss.detach()):
                        raise FloatingPointError('Non-finite GRPO loss')
                    self.accelerator.backward(loss)
                    result += loss.detach()
                del item
        if total == 0:
            # Do not let Adam momentum or weight decay update an empty batch.
            model.zero_grad(set_to_none=True)
        return result
