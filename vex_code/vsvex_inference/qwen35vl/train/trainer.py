"""Shared SFT trainer, finite-loss checks and FSDP2 checkpoint handling."""

import logging
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.distributed as dist
import transformers
from safetensors.torch import save_file


class FiniteLossTrainer(transformers.Trainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        lengths = getattr(dataset, "cached_token_lengths", None)
        if lengths is None:
            return super()._get_train_sampler(train_dataset)
        from qwenvl.data.length_sampler import GlobalLengthGroupedSampler

        if self.accelerator.split_batches or self.accelerator.dispatch_batches:
            raise ValueError("Length cache sampling requires split_batches=False and dispatch_batches=False")
        if getattr(self.args, "train_sampling_strategy", "random") != "random":
            raise ValueError("Length cache sampling requires train_sampling_strategy=random")
        sampler = GlobalLengthGroupedSampler(
            lengths, self._train_batch_size, self.args.world_size,
            seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
            window_batches=self.args.length_grouping_window,
            drop_last=self.args.dataloader_drop_last,
        )
        dataset._length_sampler = sampler
        logging.warning("Using global length grouping: %d samples, %d ranks, %d samples/rank, window=%d global batches",
                        len(lengths), self.args.world_size, self._train_batch_size, self.args.length_grouping_window)
        return sampler

    def _save(self, output_dir=None, state_dict=None):
        from peft import PeftModel
        from peft.tuners.tuners_utils import BaseTuner

        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if isinstance(model, transformers.PreTrainedModel) and not isinstance(model, PeftModel):
            output_dir = output_dir or self.args.output_dir
            # Transformers defaults to 50GB shards; keep full-model checkpoints
            # small enough to copy and load individually, including under FSDP2.
            model.save_pretrained(output_dir, state_dict=state_dict, max_shard_size="5GB")
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            elif getattr(self.data_collator, "tokenizer", None) is not None:
                self.data_collator.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, transformers.trainer.TRAINING_ARGS_NAME))
            return

        super()._save(output_dir, state_dict=state_dict)
        if not isinstance(self.model, PeftModel):
            return
        if state_dict is None:
            raise RuntimeError("FSDP2 PEFT saving requires the gathered state_dict")
        output_dir = Path(output_dir or self.args.output_dir)
        trainable_names = {
            name.replace("._checkpoint_wrapped_module", "")
            for name, param in self.model.named_parameters() if param.requires_grad
        }
        missing = trainable_names - state_dict.keys()
        if missing:
            raise RuntimeError(f"FSDP2 checkpoint is missing {len(missing)} trainable tensors: {sorted(missing)[:5]}")
        trainable_state = {
            name: state_dict[name].detach().cpu().contiguous()
            for name in sorted(trainable_names)
        }
        lora_state = {
            BaseTuner._remove_adapter_name_from_key(name, "default"): value
            for name, value in trainable_state.items() if "lora_" in name
        }
        if not lora_state:
            raise RuntimeError("PEFT checkpoint has no LoRA tensors")
        save_file(lora_state, output_dir / "adapter_model.safetensors")
        save_file(trainable_state, output_dir / "trainable_weights.safetensors")
        logging.warning("Saved %d LoRA and %d total trainable tensors to %s",
                        len(lora_state), len(trainable_state), output_dir)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.is_world_process_zero() and self.args.hdfs_output_dir:
            checkpoint = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
            env = os.environ.copy()
            # The cluster's Cruise client uses the system Python packages.
            env["PYTHONNOUSERSITE"] = "1"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[2] / "scripts/upload_checkpoint.py"),
                    str(checkpoint),
                    self.args.hdfs_output_dir,
                    "--save_total_limit",
                    str(self.args.save_total_limit or 5),
                ],
                env=env,
            )
            self._hdfs_uploads = getattr(self, "_hdfs_uploads", []) + [process]

    def wait_for_uploads(self):
        if self.is_world_process_zero():
            for process in getattr(self, "_hdfs_uploads", []):
                if process.wait() != 0:
                    raise RuntimeError("HDFS checkpoint upload failed; see the training stdout/stderr")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss = result[0] if return_outputs else result
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("Non-finite training loss; aborting instead of logging it as zero")
        return result
