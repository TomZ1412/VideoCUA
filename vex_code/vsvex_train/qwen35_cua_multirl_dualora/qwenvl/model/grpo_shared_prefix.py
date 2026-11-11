"""Differentiable, layer-local prefix branching for Qwen3.5 GRPO.

The decoder sees [prompt, response_0, ..., response_B-1] as one packed row.
Tokenwise projections/MLPs process the prompt once. Each token mixer forks its
prefix state into independent response rows, then restores the packed layout.
All temporary state lives inside one decoder call, including checkpoint replay:
no inference cache, detached tensors, or state survives a forward/backward.
"""
from types import MethodType, SimpleNamespace

import torch


class _PrefixState:
    """Small differentiable adapter for the upstream attention/cache interface."""

    def __init__(self, layer_idx):
        self.layer = SimpleNamespace(record_past=True, recurrent_states=[None])
        self.layers = {layer_idx: self.layer}
        self.conv = None
        self.key = self.value = None

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.layer.recurrent_states[0] is not None

    def update_conv_state(self, x, layer_idx, conv_kernel_size):
        if self.conv is None:
            self.conv = x[..., -(conv_kernel_size - 1):]
            return x
        return torch.cat((self.conv.expand(x.shape[0], -1, -1), x), dim=-1)

    def update_recurrent_state(self, state, layer_idx):
        self.layer.recurrent_states[0] = state

    def update(self, key, value, layer_idx):
        if self.key is None:
            self.key, self.value = key, value
            return key, value
        batch = key.shape[0]
        return (torch.cat((self.key.expand(batch, -1, -1, -1), key), dim=2),
                torch.cat((self.value.expand(batch, -1, -1, -1), value), dim=2))


def _split(x, layout):
    prefix, batch, width = layout
    return x[:, :prefix], x[:, prefix:].reshape(batch, width, x.shape[-1])


def _join(prefix, responses):
    return torch.cat((prefix, responses.reshape(1, -1, responses.shape[-1])), dim=1)


def _linear_forward(self, hidden_states, cache_params=None, attention_mask=None,
                    grpo_shared_prefix=None, **kwargs):
    if grpo_shared_prefix is None:
        return self._grpo_original_forward(hidden_states, cache_params=cache_params,
                                           attention_mask=attention_mask, **kwargs)
    if cache_params is not None:
        raise ValueError('Shared-prefix training cannot use an inference cache')
    prefix, responses = _split(hidden_states, grpo_shared_prefix)
    if responses.shape[1] < 2:
        # Upstream's single-token recurrent decode is an inference kernel.
        raise ValueError('Shared-prefix response width must be padded to at least 2')
    state = _PrefixState(self.layer_idx)
    prefix_out = self._grpo_original_forward(prefix, cache_params=state, **kwargs)
    recurrent = state.layer.recurrent_states[0]
    state.layer.recurrent_states[0] = recurrent.expand(responses.shape[0], -1, -1, -1).contiguous()
    response_out = self._grpo_original_forward(responses, cache_params=state, **kwargs)
    return _join(prefix_out, response_out)


def _attention_forward(self, hidden_states, position_embeddings, attention_mask=None,
                       past_key_values=None, grpo_shared_prefix=None, **kwargs):
    if grpo_shared_prefix is None:
        return self._grpo_original_forward(
            hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, **kwargs)
    if past_key_values is not None:
        raise ValueError('Shared-prefix training cannot use an inference cache')
    # FlashAttention's bottom-right causal mask supports Q=response, K=prompt+response.
    # No quadratic tree mask is materialized. Right-padded response positions
    # cannot influence valid positions and are excluded by the fused loss mask.
    kwargs.pop('position_ids', None)
    prefix, responses = _split(hidden_states, grpo_shared_prefix)
    split_positions = [_split(x, grpo_shared_prefix) for x in position_embeddings]
    state = _PrefixState(self.layer_idx)
    prefix_out, _ = self._grpo_original_forward(
        prefix, position_embeddings=tuple(x[0] for x in split_positions),
        attention_mask=None, past_key_values=state, **kwargs)
    response_out, _ = self._grpo_original_forward(
        responses, position_embeddings=tuple(x[1] for x in split_positions),
        attention_mask=None, past_key_values=state, **kwargs)
    return _join(prefix_out, response_out), None


def enable_shared_prefix(model):
    """Install before PEFT/FSDP wrapping; parameter names and layers stay intact."""
    language = model.model.language_model
    if language.config._attn_implementation not in ('flash_attention_2', 'flash_attention_3'):
        raise ValueError('GRPO shared-prefix training requires FlashAttention 2 or 3')
    for layer in language.layers:
        mixer = layer.linear_attn if layer.block_type == 'linear_attention' else layer.self_attn
        if hasattr(mixer, '_grpo_original_forward'):
            continue
        mixer._grpo_original_forward = mixer.forward
        mixer.forward = MethodType(
            _linear_forward if layer.block_type == 'linear_attention' else _attention_forward, mixer)


def branch_position_ids(position_ids, layout, device):
    prefix, batch, width = layout
    if position_ids is None:
        position_ids = torch.arange(prefix + batch * width, device=device)[None, None].expand(3, 1, -1)
    return torch.cat((position_ids[..., :prefix],
                      position_ids[..., prefix:prefix + width].repeat(1, 1, batch)), dim=-1)


def completion_hidden(hidden, layout):
    prefix, batch, width = layout
    branches = hidden[:, prefix:].reshape(batch, width, -1)
    # The last prompt token predicts the first token of every response.
    return torch.cat((hidden[:, prefix - 1:prefix].expand(batch, -1, -1), branches[:, :-1]), dim=1)
