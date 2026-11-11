"""Audio extensions for the Transformers Qwen3.5 Dense and MoE configs."""

from huggingface_hub.dataclasses import strict
from transformers import BertConfig, WhisperConfig
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig


class AudioBertConfig(BertConfig):
    def __init__(
        self,
        hidden_size=3072,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=12288,
        encoder_width=1280,
        query_length=1,
        cross_attention_freq=1,
        out_hidden_size=4096,
        **kwargs,
    ):
        kwargs.setdefault("add_cross_attention", True)
        kwargs.setdefault("is_decoder", True)
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            **kwargs,
        )
        self.encoder_width = encoder_width
        self.query_length = query_length
        self.cross_attention_freq = cross_attention_freq
        self.out_hidden_size = out_hidden_size


class AudioWhisperConfig(WhisperConfig):
    def __init__(self, qformer_config=None, second_per_window=0.5, second_stride=0.5,
                 num_mel_bins=128, d_model=1280, encoder_layers=32,
                 encoder_attention_heads=20, encoder_ffn_dim=5120, **kwargs):
        super().__init__(num_mel_bins=num_mel_bins, d_model=d_model,
                         encoder_layers=encoder_layers,
                         encoder_attention_heads=encoder_attention_heads,
                         encoder_ffn_dim=encoder_ffn_dim, **kwargs)
        self.qformer_config = (
            AudioBertConfig(**qformer_config)
            if isinstance(qformer_config, dict)
            else qformer_config or AudioBertConfig()
        )
        self.second_per_window = second_per_window
        self.second_stride = second_stride


def _prepare_audio_config(config):
    if isinstance(config.audio_config, dict):
        config.audio_config = AudioWhisperConfig(**config.audio_config)
    elif config.audio_config is None:
        config.audio_config = AudioWhisperConfig()


@strict
class Qwen3_5AudioConfig(Qwen3_5Config):
    sub_configs = {**Qwen3_5Config.sub_configs, "audio_config": AudioWhisperConfig}
    audio_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 248076

    def __post_init__(self, **kwargs):
        _prepare_audio_config(self)
        super().__post_init__(**kwargs)
        self.audio_config.qformer_config.out_hidden_size = self.text_config.hidden_size


@strict
class Qwen3_5MoeAudioConfig(Qwen3_5MoeConfig):
    sub_configs = {**Qwen3_5MoeConfig.sub_configs, "audio_config": AudioWhisperConfig}
    audio_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 248076

    def __post_init__(self, **kwargs):
        _prepare_audio_config(self)
        super().__post_init__(**kwargs)
        self.audio_config.qformer_config.out_hidden_size = self.text_config.hidden_size
