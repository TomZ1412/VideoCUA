"""Global micro-batch ordering; Accelerate performs the only rank sharding."""

import math
import random

from torch.utils.data import Sampler


class GlobalLengthGroupedSampler(Sampler):
    def __init__(self, lengths, batch_size, world_size, seed=42, window_batches=64, drop_last=False):
        self.lengths = list(lengths)
        if not self.lengths or any(not math.isfinite(x) or x <= 0 for x in self.lengths):
            raise ValueError("Lengths must be nonempty, finite and positive")
        if min(batch_size, world_size, window_batches) < 1:
            raise ValueError("Batch size, world size and window_batches must be positive")
        self.batch_size = batch_size
        self.world_size = world_size
        self.global_size = batch_size * world_size
        self.window_size = self.global_size * window_batches
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if drop_last and len(self.lengths) < self.global_size:
            raise ValueError("Dataset is smaller than one global batch with drop_last=True")

    def __len__(self):
        n = len(self.lengths)
        if self.drop_last:
            return n // self.global_size * self.global_size
        return math.ceil(n / self.global_size) * self.global_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        if self.drop_last:
            indices = indices[:len(self)]
        else:
            # Pad before sorting, so the tail cannot introduce an unbalanced step.
            indices += [indices[i % len(indices)] for i in range(len(self) - len(indices))]
        batches = []
        for start in range(0, len(indices), self.window_size):
            window = sorted(indices[start:start + self.window_size], key=self.lengths.__getitem__)
            for offset in range(0, len(window), self.global_size):
                group = window[offset:offset + self.global_size]
                # Equal sample counts per rank; distribute expensive samples first.
                ranks = [[] for _ in range(self.world_size)]
                costs = [0] * self.world_size
                for idx in reversed(group):
                    rank = min((r for r in range(self.world_size) if len(ranks[r]) < self.batch_size),
                               key=lambda r: costs[r])
                    ranks[rank].append(idx)
                    costs[rank] += self.lengths[idx]
                rng.shuffle(ranks)
                batches.append([idx for rank in ranks for idx in rank])
        rng.shuffle(batches)
        for batch in batches:
            yield from batch
