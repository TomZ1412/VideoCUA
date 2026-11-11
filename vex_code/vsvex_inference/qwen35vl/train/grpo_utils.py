"""Pure rollout scheduling and single-turn thinking rewards.

No model or media imports: these invariants can be tested on CPU and at 64 ranks.
"""
import math
import random
import re
from bisect import bisect_left


def final_answer(text):
    """The Qwen template already contributes the opening <think> tag."""
    if text.count('</think>') != 1:
        return None
    reasoning, answer = text.split('</think>')
    reasoning = reasoning.removeprefix('<think>').strip()
    answer = answer.strip()
    if not reasoning or not answer or re.search(r'</?think>|<tool_call>|<thinking>', reasoning + answer):
        return None
    return answer


def answer_key(text):
    text = text.strip()
    # MC answers must be a standalone option, not the first letter of a word.
    match = re.match(r'^([A-Z])(?=$|[\s.)：:])', text)
    return match.group(1) if match else text.casefold()


def rewards(text, reference, token_count, max_tokens):
    answer = final_answer(text)
    expected = answer_key(reference)
    actual = answer_key(answer) if answer else None
    # Accept an option followed by an explanation but reject e.g. "Because" for B.
    if answer and re.fullmatch(r'[A-Z]', expected):
        match = re.match(r'^([A-Z])(?=$|[\s.)：:])', answer)
        actual = match.group(1) if match else None
    # No penalty in the first half; linearly penalize excess up to -1 at the limit.
    length_reward = -min(1.0, max(0.0, (token_count - max_tokens / 2) / (max_tokens / 2)))
    return (length_reward, float(answer is not None), float(actual == expected))


def group_advantages(values):
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return [(v - mean) / (math.sqrt(variance) + 1e-4) for v in values], variance > 1e-12


def dynamic_group_valid(accuracies):
    """Keep partially-correct groups, matching the reference trainer's acc filter."""
    correct = sum(value == 1.0 for value in accuracies)
    return 0 < correct < len(accuracies)


def balance_rollout_groups(groups, ranks):
    """Assign metadata for whole prompt groups, with 10% locality slack.

    Balance completion counts first (FSDP pads to the maximum), then actual
    prompt-token work. Prefer the source rank when its projected token load is
    within 10% of the least-loaded eligible rank. Repeated dataset indices are
    bundled too, so padded/supplemental candidates cannot duplicate a video's
    training media across ranks. Such bundles can make counts less even.
    """
    if ranks < 1:
        raise ValueError('ranks must be positive')
    bundles = {}
    for group in groups:
        bundles.setdefault(group['index'], []).append(group)
    bins, counts, costs = [[] for _ in range(ranks)], [0] * ranks, [0] * ranks

    def work(bundle):
        return sum(g['prompt_length'] * g['count'] for g in bundle)

    for bundle in sorted(bundles.values(), key=work, reverse=True):
        cost = work(bundle)
        eligible = [r for r in range(ranks) if counts[r] == min(counts)]
        best = min(costs[r] + cost for r in eligible)
        eligible = [r for r in eligible if costs[r] + cost <= best * 1.1]
        # Saving an entire local payload is preferable to an arbitrary tie.
        rank = min(eligible, key=lambda r: (
            -sum(g['count'] for g in bundle if g['source'] == r), costs[r], r))
        bins[rank].extend(bundle)
        counts[rank] += sum(g['count'] for g in bundle)
        costs[rank] += cost
    return bins


def dynamic_candidate_count(target, accept_rate, oversample_factor=1.2, max_factor=8.0):
    """Reference-style base + estimated deficit, bounded for long-video memory."""
    if not math.isfinite(accept_rate) or not 0 <= accept_rate <= 1:
        raise ValueError('accept_rate must be in [0, 1]')
    if not math.isfinite(max_factor) or max_factor < 1:
        raise ValueError('dynamic_sampling_max_factor must be finite and >= 1')
    cap = max(target, math.floor(target * max_factor))
    if accept_rate == 0:
        return cap
    extra = (target / accept_rate - target) * oversample_factor
    return cap if extra >= cap - target else target + math.ceil(extra)


class NodePromptSampler:
    """Pick similar-length *unique prompts*, then assign one group to each node.

    With 64 ranks, local_size=8, B=1, A=1, N=8, each step chooses 8
    prompts, one per node. Each node keeps its 8 completions on one rank;
    idle ranks perform zero-advantage padding for FSDP. No Accelerate
    rank-sharding must be applied again.
    """
    def __init__(self, lengths, rank, world_size, local_size, batch_size,
                 accumulation_steps, num_generations, seed=42, window_batches=64,
                 oversample_factor=1.0, modalities=None, initial_accept_rate=None,
                 max_oversample_factor=8.0):
        if min(world_size, local_size, batch_size, accumulation_steps, num_generations, window_batches) < 1:
            raise ValueError('All scheduling sizes must be positive')
        if world_size % local_size or not 0 <= rank < world_size:
            raise ValueError('Require homogeneous nodes and a valid rank')
        budget = local_size * batch_size * accumulation_steps
        if budget % num_generations:
            raise ValueError('local_size * bs * accum_steps must be divisible by num_generations')
        if not math.isfinite(oversample_factor) or oversample_factor < 1:
            raise ValueError('oversample_factor must be finite and >= 1')
        self.lengths = list(lengths)
        if not self.lengths or any(not math.isfinite(x) or x <= 0 for x in self.lengths):
            raise ValueError('Lengths must be positive and finite')
        modalities = list(modalities) if modalities is not None else [None] * len(self.lengths)
        if len(modalities) != len(self.lengths):
            raise ValueError('One modality signature is required per sample')
        self.buckets = {}
        for index, modality in enumerate(modalities):
            self.buckets.setdefault(modality, []).append(index)
        self.node = rank // local_size
        self.nodes = world_size // local_size
        self.target_prompts_per_node = budget // num_generations
        self.oversample_factor = oversample_factor
        self.max_oversample_factor = max_oversample_factor
        self.adaptive = initial_accept_rate is not None
        self.prompts_per_node = math.ceil(self.target_prompts_per_node * oversample_factor)
        if self.adaptive:
            self.update_acceptance(initial_accept_rate)
        # Epoch length follows the nominal base plan. Supplemental candidates
        # do not change Trainer's step count or prefetch future media batches.
        self.global_prompts = self.nodes * (self.target_prompts_per_node if self.adaptive else self.prompts_per_node)
        self.seed, self.epoch, self.window_batches = seed, 0, window_batches
        self.plan_mode = False
        self._plans = {}

    def __len__(self):
        return sum(math.ceil(len(bucket) / self.global_prompts) for bucket in self.buckets.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_acceptance(self, accept_rate):
        self.prompts_per_node = dynamic_candidate_count(
            self.target_prompts_per_node, accept_rate, self.oversample_factor, self.max_oversample_factor)

    def _supplement(self, step, pool, pool_lengths, prompts_per_node=None):
        count = self.nodes * (self.prompts_per_node if prompts_per_node is None else prompts_per_node)
        result = list(step)
        seen = set(result)
        center = sum(self.lengths[i] for i in step) / len(step)
        right = bisect_left(pool_lengths, center)
        left = right - 1
        while len(result) < count and (left >= 0 or right < len(pool)):
            if left >= 0 and (right == len(pool) or center - pool_lengths[left] <= pool_lengths[right] - center):
                index = pool[left]
                left -= 1
            else:
                index = pool[right]
                right += 1
            if index not in seen:
                result.append(index)
                seen.add(index)
        # Small modality buckets may need repeats, just like padded base plans.
        if len(result) < count:
            result += [pool[i % len(pool)] for i in range(count - len(result))]
        # Preserve the base and supplemental prefix when a later acceptance
        # estimate grows the plan. Resorting would move prefetched prompts
        # between nodes / TP leaders and invalidate their media caches.
        return result

    def _epoch_plans(self, epoch):
        if epoch in self._plans:
            return self._plans[epoch]
        rng = random.Random(self.seed + epoch)
        steps = []
        window = self.global_prompts * self.window_batches
        # FSDP collectives must visit the same visual/audio branches, not only
        # have matching iteration counts. Keep each global step homogeneous.
        for bucket in self.buckets.values():
            indices = list(bucket)
            rng.shuffle(indices)
            pool = sorted(indices, key=self.lengths.__getitem__)
            pool_lengths = [self.lengths[i] for i in pool]
            count = math.ceil(len(indices) / self.global_prompts) * self.global_prompts
            indices += [indices[i % len(indices)] for i in range(count - len(indices))]
            for start in range(0, len(indices), window):
                ordered = sorted(indices[start:start + window], key=self.lengths.__getitem__)
                steps.extend((ordered[i:i + self.global_prompts], pool, pool_lengths)
                             for i in range(0, len(ordered), self.global_prompts))
        rng.shuffle(steps)
        self._plans[epoch] = steps
        for stale in list(self._plans):
            if stale < epoch - 1:
                del self._plans[stale]
        return steps

    def resolve_plan(self, key):
        epoch, offset = key
        step, pool, pool_lengths = self._epoch_plans(epoch)[offset]
        if self.adaptive:
            step = self._supplement(step, pool, pool_lengths)
        return step[self.node::self.nodes]

    def __iter__(self):
        epoch = self.epoch
        plans = self._epoch_plans(epoch)
        for offset in range(len(plans)):
            key = (epoch, offset)
            if self.plan_mode:
                following = (epoch, offset + 1) if offset + 1 < len(plans) else (epoch + 1, 0)
                yield {'plan_key': key, 'next_plan_key': following}
            else:
                yield self.resolve_plan(key)
