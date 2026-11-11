"""Prepare an immutable merged base before starting distributed training."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def prepare_model(model, model_base, lora_ckpt, output_dir):
    model = Path(model).resolve()
    fallback = Path(model_base).resolve() if model_base else None
    adapter = Path(lora_ckpt).resolve() if lora_ckpt else None
    if (model / "adapter_config.json").is_file():
        if adapter is not None and adapter != model:
            raise ValueError("model is already an adapter; do not supply a different lora_ckpt")
        adapter = model
        saved_base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
        model = fallback or Path(saved_base).resolve()
    if adapter is None:
        return model

    sources = list(dict.fromkeys([model, adapter] + ([fallback] if fallback else [])))
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(source)
    request = dict(model=str(model), model_base=str(fallback), adapter=str(adapter), stamps=[
        [(str(p.relative_to(source)), p.stat().st_size, p.stat().st_mtime_ns)
         for p in sorted(source.rglob("*"))
         if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".jinja")]
        for source in sources
    ])
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    output = Path(output_dir).resolve() / "base"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Serialize workers on a node, including nodes sharing the output filesystem.
    with output.with_name(".base.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = output / "train_merge.json"
        if marker.is_file() and json.loads(marker.read_text()).get("signature") == signature:
            return output
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Existing base has different or unverified sources; use a new output_dir: {output}")
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".merge-") as stage:
            command = [sys.executable, str(ROOT / "scripts/merge_base_model.py"),
                       "--model_name_or_path", str(model), "--lora_ckpt", str(adapter),
                       "--output_dir", stage]
            if fallback:
                command.extend(["--model_base", str(fallback)])
            subprocess.run(command, check=True)
            (Path(stage) / "train_merge.json").write_text(json.dumps(dict(signature=signature, **request), indent=2) + "\n")
            if output.exists():
                output.rmdir()
            Path(stage).rename(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_base")
    parser.add_argument("--lora_ckpt")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    prepare_model(args.model, args.model_base, args.lora_ckpt, args.output_dir)


def prepare_generation_model(model, model_base, output_dir):
    """A local asset overlay for vLLM; never modify the source checkpoint.

    Trainer checkpoints often omit image/video processor configs. The HF training
    path can load those from model_base, while vLLM needs one complete directory.
    """
    model, fallback = Path(model).resolve(), Path(model_base).resolve()
    target = Path(output_dir).resolve() / 'generation_model'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name('.generation_model.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target.mkdir(exist_ok=True)
        for source in (model, fallback):
            for path in source.iterdir():
                if source != model and not (path.suffix == '.jinja' or path.name.startswith(
                    ('tokenizer', 'preprocessor', 'video_preprocessor', 'audio_processor'))):
                    continue
                dest = target / path.name
                if dest.is_symlink() and dest.resolve() != path.resolve() and source == model:
                    raise ValueError(f'Generation overlay refers to a different checkpoint: {dest}')
                if not dest.exists():
                    dest.symlink_to(path, target_is_directory=path.is_dir())
    return target
