"""Fused DAPO/GRPO with token importance correction for vLLM sampling.

A rollout is consumed once before an optimizer update. Thus the old training
policy equals the detached current policy, as in TRL's single-iteration path.
The vLLM behavior distribution is a separate denominator for importance sampling;
using it for PPO clipping would incorrectly clip top-k/top-p sampling differences.
"""
import torch
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss


class GRPOLoss(torch.nn.Module):
    def __init__(self, temperature=1.0, epsilon=.2, importance_cap=2.0, compiled=True,
                 importance_sampling_level='token', loss_type='dapo'):
        super().__init__()
        if loss_type not in ('dapo', 'grpo'):
            raise ValueError('loss_type must be dapo or grpo')
        self.loss_type = loss_type
        if importance_sampling_level not in ('token', 'sequence'):
            raise ValueError('importance_sampling_level must be token or sequence')
        self.importance_sampling_level = importance_sampling_level
        self.temperature, self.importance_cap = temperature, importance_cap
        # Use Liger's local GRPO reduction as a token-chunked primitive for both
        # objectives. Scaling below selects the local mean; the trainer handles
        # global normalization across all rollouts in the optimizer step. Native
        # Liger DAPO can otherwise all-reduce inside unequal token chunk loops.
        self.fused = LigerFusedLinearGRPOLoss(
            beta=0, use_ref_model=False, loss_type='grpo', temperature=temperature,
            epsilon_low=epsilon, epsilon_high=epsilon, compiled=compiled, chunk_size=128,
        )

    def forward(self, hidden, weight, selected_token_ids, attention_mask,
                advantages, sampling_per_token_logps):
        weight = weight.to(hidden.dtype)
        with torch.no_grad():
            chunks = []
            # Bound vocabulary-sized temporaries independently of context length.
            for start in range(0, hidden.shape[1], 128):
                logits = torch.nn.functional.linear(hidden[:, start:start + 128], weight).float()
                logits.div_(self.temperature)
                tokens = selected_token_ids[:, start:start + 128, None]
                logps = logits.gather(-1, tokens).squeeze(-1) - logits.logsumexp(-1)
                chunks.append(logps)
            policy_logps = torch.cat(chunks, dim=1)
            correction = (policy_logps - sampling_per_token_logps).exp().clamp(max=self.importance_cap)
            if self.importance_sampling_level == 'sequence':
                # GSPO ratio: exp(mean_valid(logp - old_logp)). In this trainer
                # old_logp = logp.detach() (one update per rollout), so r=1 and
                # clipping is inactive. For L=-A*r*mean_valid(correction), its
                # value and first derivative equal the token surrogate with
                # mean_valid(correction) broadcast to every token. This preserves
                # the exact sequence gradient while keeping vocabulary logits
                # token-chunked. Passing 'sequence' to Liger after flattening
                # would instead treat every token as a separate sequence.
                # This reduction must be revisited if rollout reuse / stored
                # old-policy logps or nonzero KL are introduced.
                mask = attention_mask.bool()
                mean_correction = correction.masked_fill(~mask, 0).sum(-1, keepdim=True)
                mean_correction /= attention_mask.sum(-1, keepdim=True).clamp(min=1)
                correction = mean_correction.expand_as(correction)
        batch, tokens, width = hidden.shape
        # Liger chunks its *batch* dimension. A [1, 16384, H] input would still
        # allocate 16384*vocab logits. Treat each token as a length-one sequence
        # and adjust advantages to undo its mean over batch*tokens entries.
        if self.loss_type == 'dapo':
            # Local token mean; training_step weights it by valid token count.
            scales = batch * tokens / attention_mask.sum().clamp(min=1)
        else:
            # Original mean of per-completion token means.
            scales = tokens / attention_mask.sum(-1).clamp(min=1)
        token_advantages = (advantages * scales)[:, None].expand(batch, tokens).reshape(-1)
        return self.fused(
            hidden.reshape(-1, 1, width), weight, selected_token_ids.reshape(-1, 1),
            attention_mask.reshape(-1, 1), token_advantages,
            old_per_token_logps=policy_logps.reshape(-1, 1),
            vllm_is_ratio=correction.reshape(-1, 1),
        )
