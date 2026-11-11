"""Two-adapter CUA GRPO on the native Qwen3.5 model and FSDP2 path."""

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from safetensors.torch import save_file

from qwenvl.data.cua_data import CUAModalitySampler
from qwenvl.data.data_qwen import multimodal_token_type_ids
from qwenvl.train.trainer import FiniteLossTrainer


GUIDE_PREFIX = "\nHere is a guide of the task which you can use to help the model to generate the answer:\n"
LEGACY_RETURNS_PER_SAMPLE = 3  # Original CUA uses num_return_sequences=3, not beam search.
MODEL_KEYS = (
    "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values",
    "image_grid_thw", "pixel_values_videos", "video_grid_thw", "audio_feature",
)


def _json_objects(text):
    """Balanced-brace matches, equivalent to the original recursive regex."""
    openings = []
    candidates = []
    for index, char in enumerate(text):
        if char == "{":
            openings.append(index)
        elif char == "}" and openings:
            candidates.append((openings.pop(), index + 1))
    matches = []
    next_start = 0
    for start, end in sorted(candidates):
        if start >= next_start:
            matches.append(text[start:end])
            next_start = end
    return matches


def action_score(prediction, reference):
    # Preserve the original CUA JSON extraction and action scoring semantics.
    predicted = _json_objects(prediction)
    expected = [json.loads(item) for item in _json_objects(reference)]
    expected = [item for item in expected if "arguments" in item]
    if not predicted or len(predicted) > 3:
        return 0.0
    score = 0.0
    count = 0
    for raw_candidate, target in zip(predicted, expected):
        count += 1
        try:
            candidate = json.loads(raw_candidate)
        except json.JSONDecodeError:
            continue
        if "arguments" not in candidate:
            continue
        args = candidate["arguments"]
        correct = target["arguments"]
        if not isinstance(args, dict) or args.get("action") != correct.get("action"):
            continue
        if "coordinate" in args:
            try:
                x, y = [int(value) for value in args["coordinate"]]
                distance = abs(x - correct["coordinate"][0]) + abs(y - correct["coordinate"][1])
                score += math.exp(-distance / 20)
            except (ValueError, TypeError, KeyError, IndexError):
                continue
        elif "text" in args and "text" in correct and args["text"] == correct["text"]:
            score += 1.0
        elif "keys" in args and "keys" in correct and args["keys"] == correct["keys"]:
            score += 1.0
        elif all(key in correct and correct[key] == value for key, value in args.items()):
            score += 1.0
    count += max(0, len(expected) - len(predicted)) * 0.5
    return score / count


def group_advantages(rewards, use_rloo):
    values = torch.tensor(rewards, dtype=torch.float32)
    if len(values) < 2:
        return torch.zeros_like(values)
    if use_rloo:
        return (len(values) * values - values.sum()) / (len(values) - 1)
    return values - values.mean()


class CUADualLoRATrainer(FiniteLossTrainer):
    def __init__(self, *args, **kwargs):
        config = kwargs["args"]
        if config.per_device_train_batch_size != 1 or config.gradient_accumulation_steps != 1:
            raise ValueError("CUA GRPO currently requires --bs 1 and --accum_steps 1")
        if config.grpo_num_sample < 2 or config.action_samples < 1:
            raise ValueError("CUA GRPO requires at least two plans and one action sample")
        if config.planner_type != "plan" or config.reward_func not in ("ppl", "sample"):
            raise ValueError("CUA GRPO requires --planner_type plan and --reward_func ppl or sample")
        if config.adapter_update not in ("both", "plan", "action"):
            raise ValueError("adapter_update must be both, plan, or action")
        super().__init__(*args, **kwargs)
        if not self.is_fsdp_enabled or self.accelerator.state.fsdp_plugin.fsdp_version != 2:
            raise ValueError("CUA GRPO requires FSDP2")
        self.model_accepts_loss_kwargs = False
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.accelerator.split_batches or self.accelerator.dispatch_batches:
            raise ValueError("CUA modality grouping requires split_batches=False and dispatch_batches=False")
        return CUAModalitySampler(
            dataset.audio_modalities, self.args.world_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed)

    def _set_adapter(self, name):
        peft = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        peft.set_adapter(name)
        for param_name, param in peft.named_parameters():
            if "lora_" in param_name:
                # Keep optimizer/FSDP trainability stable across adapter switches.
                # Inactive adapters are absent from the forward graph.
                param.requires_grad_(
                    ".adapter_1." in param_name or ".adapter_2." in param_name)

    def _inputs(self, item):
        return {key: value for key in MODEL_KEYS if (value := item.get(key)) is not None}

    def _generate(self, model, item, adapter, do_sample, max_new_tokens):
        self._set_adapter(adapter)
        inputs = self._inputs(item)
        with torch.no_grad():
            output = model.generate(
                **inputs, do_sample=do_sample, top_p=0.9 if do_sample else 1.0,
                temperature=1.0, max_new_tokens=max_new_tokens, num_beams=1,
                use_cache=True,
            )
        prompt_size = inputs["input_ids"].shape[1]
        ids = output[:, prompt_size:]
        text = self.processing_class.decode(
            ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if ids.shape[1] == 0:
            raise RuntimeError("Model returned an empty CUA completion")
        return text, ids

    def _action_prompt(self, action):
        ids = action["input_ids"]
        labels = action["labels"]
        start_id = self.processing_class.convert_tokens_to_ids("<|im_start|>")
        turns = (ids[0] == start_id).nonzero(as_tuple=False)
        if len(turns) == 0:
            raise ValueError("Action sample has no assistant turn")
        last_turn = int(turns[-1, 0])
        positions = (labels[0, last_turn:] != -100).nonzero(as_tuple=False)
        if len(positions) == 0:
            raise ValueError("Final CUA action has no supervised tokens")
        answer_start = last_turn + int(positions[0, 0])
        prompt = dict(self._inputs(action))
        prompt["input_ids"] = ids[:, :answer_start]
        prompt["attention_mask"] = torch.ones_like(prompt["input_ids"])
        prompt["mm_token_type_ids"] = multimodal_token_type_ids(
            prompt["input_ids"], self.processing_class)
        return prompt, ids[:, answer_start:]

    def _with_guide(self, action_prompt, description):
        ids = action_prompt["input_ids"]
        end_token = self.processing_class.convert_tokens_to_ids("<|im_end|>")
        ends = (ids[0] == end_token).nonzero(as_tuple=False)
        if len(ends) < 2:
            raise ValueError("Cannot locate the end of the first user turn")
        insert_at = int(ends[1, 0])
        guide = self.processing_class(
            GUIDE_PREFIX + description, add_special_tokens=False,
            return_tensors="pt")["input_ids"].to(ids.device)
        guided = dict(action_prompt)
        guided["input_ids"] = torch.cat((ids[:, :insert_at], guide, ids[:, insert_at:]), dim=1)
        guided["attention_mask"] = torch.ones_like(guided["input_ids"])
        guided["mm_token_type_ids"] = multimodal_token_type_ids(
            guided["input_ids"], self.processing_class)
        return guided

    def _logps(self, model, prompt, completion):
        if completion.shape[1] < 1:
            raise ValueError("CUA completion must be nonempty")
        inputs = dict(prompt)
        inputs["input_ids"] = torch.cat((prompt["input_ids"], completion), dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["mm_token_type_ids"] = multimodal_token_type_ids(
            inputs["input_ids"], self.processing_class)
        return model(**inputs, cua_targets=completion)

    def _ppl_nll(self, model, action, description=None):
        full = self._inputs(action)
        full["attention_mask"] = torch.ones_like(full["input_ids"])
        full["mm_token_type_ids"] = multimodal_token_type_ids(
            full["input_ids"], self.processing_class)
        labels = action["labels"]
        if description is not None:
            end_token = self.processing_class.convert_tokens_to_ids("<|im_end|>")
            ends = (full["input_ids"][0] == end_token).nonzero(as_tuple=False)
            if len(ends) < 2:
                raise ValueError("Cannot locate the end of the first user turn in action PPL input")
            insert_at = int(ends[1, 0])
            guided = self._with_guide(full, description)
            added = guided["input_ids"].shape[1] - full["input_ids"].shape[1]
            labels = torch.cat((labels[:, :insert_at],
                                labels.new_full((1, added), -100),
                                labels[:, insert_at:]), dim=1)
            full = guided
        self._set_adapter("adapter_1")
        with torch.no_grad(), self.compute_loss_context_manager():
            return -model(**full, cua_labels=labels).mean().item()

    def _policy_loss(self, model, prompt, completion, advantage, adapter, reference=None):
        ref_logps = None
        if reference is not None:
            self._set_adapter(reference)
            with torch.no_grad(), self.compute_loss_context_manager():
                ref_logps = self._logps(model, prompt, completion)
        self._set_adapter(adapter)
        with self.compute_loss_context_manager():
            logps = self._logps(model, prompt, completion)
            ratio = torch.exp(logps - logps.detach())
            loss = -float(advantage) * ratio
            if ref_logps is not None:
                delta = ref_logps - logps
                kl = torch.exp(delta) - delta - 1
                loss = loss + self.args.cua_kl_beta * kl
            loss = loss.mean()
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError("Non-finite CUA policy loss")
        return loss

    def _rollouts(self, model, example, deterministic=False):
        plan = self._inputs(example["plan"])
        action, target = self._action_prompt(example["action"])
        count = 1 if deterministic else self.args.grpo_num_sample * LEGACY_RETURNS_PER_SAMPLE
        self.model.eval()
        descriptions, plan_ids, guided = [], [], []
        for _ in range(count):
            text, ids = self._generate(
                model, plan, "adapter_2", not deterministic, self.args.cua_max_plan_tokens)
            descriptions.append(text)
            plan_ids.append(ids)
            guided.append(self._with_guide(action, text))
        no_desc_nll, with_desc_nll = None, []
        action_rollouts = []
        if self.args.reward_func == "ppl":
            no_desc_nll = self._ppl_nll(model, example["action"])
            with_desc_nll = [
                self._ppl_nll(model, example["action"], description)
                for description in descriptions
            ]
            rewards = [no_desc_nll - value for value in with_desc_nll]
        else:
            rewards = []
            for prompt in guided:
                scored = []
                for _ in range(self.args.action_samples * LEGACY_RETURNS_PER_SAMPLE):
                    prediction, ids = self._generate(
                        model, prompt, "adapter_1", not deterministic, self.args.cua_max_action_tokens)
                    scored.append(action_score(prediction, example["reference"]))
                    action_rollouts.append((prompt, ids, scored[-1]))
                rewards.append(sum(scored) / len(scored))
        return plan, action, target, guided, descriptions, plan_ids, rewards, no_desc_nll, with_desc_nll, action_rollouts

    def training_step(self, model, inputs, num_items_in_batch=None):
        example = self._prepare_inputs(inputs)
        plan, action, target, guided, descriptions, plan_ids, rewards, no_desc, with_desc, action_rollouts = (
            self._rollouts(model, example))
        model.train()
        advantages = group_advantages(rewards, self.args.use_rloo).tolist()
        planned = len(plan_ids) if self.args.adapter_update in ("both", "plan") else 0
        actions = (len(guided) if self.args.reward_func == "ppl" else len(action_rollouts))
        actions = actions if self.args.adapter_update in ("both", "action") else 0
        steps = planned + actions
        if steps == 0:
            raise RuntimeError("No CUA policy updates were produced")
        result = torch.zeros((), device=self.accelerator.device)

        def update(loss):
            nonlocal result
            scaled = loss / steps
            self.accelerator.backward(scaled)
            result += scaled.detach()

        if self.args.adapter_update in ("both", "plan"):
            for ids, advantage in zip(plan_ids, advantages):
                update(self._policy_loss(
                    model, plan, ids, advantage, "adapter_2", "adapter_2_ref"))
        if self.args.adapter_update in ("both", "action"):
            if self.args.reward_func == "ppl":
                weights = torch.softmax(
                    torch.tensor(advantages, dtype=torch.float32) / 0.2, dim=0).tolist()
                for prompt, weight in zip(guided, weights):
                    update(self._policy_loss(
                        model, prompt, target, weight, "adapter_1"))
            else:
                for group in range(len(guided)):
                    sampled = action_rollouts[
                        group * self.args.action_samples * LEGACY_RETURNS_PER_SAMPLE:
                        (group + 1) * self.args.action_samples * LEGACY_RETURNS_PER_SAMPLE]
                    action_adv = group_advantages(
                        [row[2] for row in sampled], self.args.use_rloo).tolist()
                    for (prompt, ids, _), advantage in zip(sampled, action_adv):
                        update(self._policy_loss(
                            model, prompt, ids, advantage, "adapter_1", "adapter_1_ref"))
        if no_desc is not None:
            values = torch.tensor(with_desc, dtype=torch.float32)
            gaps = torch.tensor(rewards, dtype=torch.float32)
            self.log({
                "ppl_desc/no_desc_nll": no_desc,
                "ppl_desc/no_desc_reward": -no_desc,
                "ppl_desc/with_desc_nll_mean": values.mean().item(),
                "ppl_desc/with_desc_nll_std": values.std(unbiased=False).item(),
                "ppl_desc/with_desc_reward_mean": -values.mean().item(),
                "ppl_desc/gap_mean": gaps.mean().item(),
                "ppl_desc/gap_std": gaps.std(unbiased=False).item(),
                "ppl_desc/gap_min": gaps.min().item(),
                "ppl_desc/gap_max": gaps.max().item(),
                "ppl_desc/distinct_description_ratio": len(set(descriptions)) / len(descriptions),
            })
        self.log({"cua/reward_mean": sum(rewards) / len(rewards),
                  "cua/plan_advantage_std": torch.tensor(advantages).std(unbiased=False).item()})
        return result

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="validation"):
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            return {}
        model = self.model
        was_training = model.training
        model.eval()
        rank = dist.get_rank()
        world = dist.get_world_size()
        groups = []
        for modality in (False, True):
            indices = [index for index, value in enumerate(dataset.audio_modalities)
                       if value == modality]
            for start in range(0, len(indices), world):
                chunk = indices[start:start + world]
                groups.append((chunk + [chunk[0]] * (world - len(chunk)), len(chunk)))
        hit = count = nonempty = zero = errors = chars = conditioned = 0.0
        for indices, valid_count in groups:
            example = self._prepare_inputs(dataset[indices[rank]])
            plan = self._inputs(example["plan"])
            action, _ = self._action_prompt(example["action"])
            prompt = self._with_guide(
                action, self._generate(model, plan, "adapter_2", False,
                                       512)[0])
            prediction, _ = self._generate(
                model, prompt, "adapter_1", False, 512)
            if rank < valid_count:
                try:
                    score = action_score(prediction, example["reference"])
                except (TypeError, ValueError, KeyError, ZeroDivisionError, json.JSONDecodeError):
                    score = 0.0
                    errors += 1
                hit += score
                count += 1
                nonempty += bool(prediction)
                zero += score == 0.0
                chars += len(prediction)
                conditioned += 1
        stats = torch.tensor([hit, count, nonempty, zero, errors, chars, conditioned],
                             dtype=torch.float64, device=self.accelerator.device)
        dist.all_reduce(stats)
        denominator = stats[1].clamp(min=1)
        result = {
            f"{metric_key_prefix}/score": 100 * (stats[0] / denominator).item(),
            f"{metric_key_prefix}/sample_count": stats[1].item(),
            f"{metric_key_prefix}/nonempty_prediction_rate": (stats[2] / denominator).item(),
            f"{metric_key_prefix}/zero_score_rate": (stats[3] / denominator).item(),
            f"{metric_key_prefix}/scoring_error_rate": (stats[4] / denominator).item(),
            f"{metric_key_prefix}/avg_prediction_chars": (stats[5] / denominator).item(),
            f"{metric_key_prefix}/plan_conditioned_rate": (stats[6] / denominator).item(),
        }
        self.log(result)
        model.train(was_training)
        return result

    def _save_checkpoint(self, model, trial):
        if self.args.do_validation:
            self.evaluate()
        return super()._save_checkpoint(model, trial)

    def _save(self, output_dir=None, state_dict=None):
        if state_dict is None:
            raise RuntimeError("CUA FSDP2 saving requires a gathered state dict")
        path = Path(output_dir or self.args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        peft = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        for adapter in ("adapter_1", "adapter_2"):
            selected = {
                name.replace("._checkpoint_wrapped_module", "").replace(f".{adapter}.", "."):
                value.detach().cpu().contiguous()
                for name, value in state_dict.items()
                if "lora_" in name and f".{adapter}." in name
            }
            if not selected:
                raise RuntimeError(f"Checkpoint contains no weights for {adapter}")
            adapter_dir = path / adapter
            adapter_dir.mkdir(exist_ok=True)
            peft.peft_config[adapter].save_pretrained(adapter_dir)
            save_file(selected, adapter_dir / "adapter_model.safetensors")
        if self.processing_class is not None:
            self.processing_class.save_pretrained(path)
        torch.save(self.args, os.path.join(path, transformers.trainer.TRAINING_ARGS_NAME))
