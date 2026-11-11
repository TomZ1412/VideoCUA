"""Qwen3.5 multimodal models with a Whisper and Q-Former audio branch.

The backbones use upstream Transformers 5.17 models. This module provides audio
token replacement and the GRPO loss entry point for both Dense and MoE; optional
layer-local prefix branching lives in grpo_shared_prefix.py.
"""

import os

import torch

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from qwenvl.model.configuration_audio import Qwen3_5AudioConfig, Qwen3_5MoeAudioConfig
from qwenvl.model.modeling_whisper import WhisperEncoder


class _AudioForwardMixin:
    def prepare_inputs_for_generation(self, *args, audio_feature=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        # Custom audio features, like upstream visual inputs, belong to prefill.
        if kwargs.get("is_first_iteration", False) or not kwargs.get("use_cache", True):
            inputs["audio_feature"] = audio_feature
        return inputs

    def __init__(self, config):
        super().__init__(config)
        self.model.audio = WhisperEncoder(config.audio_config)
        self._no_split_modules = [*self._no_split_modules, "WhisperEncoderLayer"]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        logits_to_keep=0,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        audio_feature=None,
        grpo_loss=None,
        grpo_targets=None,
        grpo_shared_prefix=None,
        cua_targets=None,
        cua_labels=None,
        **kwargs,
    ):
        if grpo_shared_prefix is not None:
            from qwenvl.model.grpo_shared_prefix import branch_position_ids
            if grpo_loss is None:
                raise ValueError("Shared-prefix layout is only supported for GRPO training")
            if position_ids is None:
                position_ids = self.model.compute_3d_position_ids(
                    input_ids=input_ids, inputs_embeds=inputs_embeds,
                    image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids)
            position_ids = branch_position_ids(position_ids, grpo_shared_prefix, input_ids.device)
        if self.training and input_ids is not None and os.environ.get("LOCAL_RANK") == "0":
            video_tokens = (input_ids == self.config.video_token_id).sum().item()
            audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
            print(
                f"LOCAL_RANK 0 video tokens: {video_tokens}, audio tokens: {audio_tokens}",
                flush=True,
            )
        if audio_feature is not None:
            if input_ids is None or inputs_embeds is not None:
                raise ValueError("Audio input requires input_ids and no inputs_embeds")
            inputs_embeds = self.get_input_embeddings()(input_ids)
            audio_embeds = self.model.audio(audio_feature.to(self.model.audio.dtype))
            audio_embeds = audio_embeds.to(inputs_embeds.dtype)
            audio_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            if audio_mask.sum().item() != audio_embeds.shape[0] * audio_embeds.shape[1]:
                raise ValueError(
                    f"Audio tokens ({audio_mask.sum().item()}) do not match encoded frames "
                    f"({audio_embeds.shape[0] * audio_embeds.shape[1]})"
                )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
            if position_ids is None:
                position_ids = self.model.compute_3d_position_ids(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    mm_token_type_ids=mm_token_type_ids,
                )
            input_ids = None
        if cua_targets is not None or cua_labels is not None:
            # Return only selected-token log probabilities. Building full
            # [sequence, vocabulary] logits for video prompts is prohibitive.
            if (cua_targets is None) == (cua_labels is None):
                raise ValueError("Specify exactly one of cua_targets and cua_labels")
            if cua_targets is not None and (
                cua_targets.ndim != 2 or cua_targets.shape[0] != 1 or cua_targets.shape[1] < 1
            ):
                raise ValueError("cua_targets must contain one nonempty completion")
            outputs = self.model(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, position_ids=position_ids,
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids, use_cache=False,
            )
            from torch.distributed.tensor import DTensor
            weight = self.lm_head.weight
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            if cua_labels is None:
                hidden = outputs.last_hidden_state[:, -cua_targets.shape[1] - 1:-1]
                selected_ids = cua_targets
            else:
                if cua_labels.shape != outputs.last_hidden_state.shape[:2]:
                    raise ValueError("cua_labels must align with input_ids")
                mask = cua_labels[:, 1:] != -100
                if not mask.any():
                    raise ValueError("cua_labels contain no supervised tokens")
                hidden = outputs.last_hidden_state[:, :-1][mask].unsqueeze(0)
                selected_ids = cua_labels[:, 1:][mask].unsqueeze(0)
            weight = weight.to(hidden.dtype)
            chunks = []
            for start in range(0, hidden.shape[1], 64):
                logits = torch.nn.functional.linear(hidden[:, start:start + 64], weight).float()
                targets = selected_ids[:, start:start + 64, None]
                chunks.append(logits.gather(-1, targets).squeeze(-1) - logits.logsumexp(-1))
            return torch.cat(chunks, dim=1)
        if grpo_loss is not None:
            # Execute inside the root FSDP forward so its parameter and backward
            # hooks remain active. Only completion hidden states enter fused CE.
            outputs = self.model(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                attention_mask=({'full_attention': None, 'linear_attention': None}
                                if grpo_shared_prefix is not None else attention_mask),
                position_ids=position_ids,
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids, use_cache=False,
                **({'grpo_shared_prefix': grpo_shared_prefix} if grpo_shared_prefix is not None else {}),
            )
            from torch.distributed.tensor import DTensor
            weight = self.lm_head.weight
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            count = grpo_targets["selected_token_ids"].shape[1]
            if grpo_shared_prefix is None:
                hidden = outputs.last_hidden_state[:, -count - 1:-1, :]
            else:
                from qwenvl.model.grpo_shared_prefix import completion_hidden
                hidden = completion_hidden(outputs.last_hidden_state, grpo_shared_prefix)
            return grpo_loss(hidden, weight, **grpo_targets)[0]
        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            logits_to_keep=logits_to_keep,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )


class Qwen3_5AudioForConditionalGeneration(_AudioForwardMixin, Qwen3_5ForConditionalGeneration):
    config_class = Qwen3_5AudioConfig


class Qwen3_5MoeAudioForConditionalGeneration(_AudioForwardMixin, Qwen3_5MoeForConditionalGeneration):
    config_class = Qwen3_5MoeAudioConfig


def model_class_for_config(config):
    if config.model_type == "qwen3_5":
        return Qwen3_5AudioForConditionalGeneration
    if config.model_type == "qwen3_5_moe":
        return Qwen3_5MoeAudioForConditionalGeneration
    raise ValueError(f"Unsupported model_type: {config.model_type}")
