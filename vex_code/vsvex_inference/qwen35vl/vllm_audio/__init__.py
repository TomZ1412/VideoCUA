"""vLLM plugin, discovered in the API server and spawned GPU workers."""
def ensure_plugin_installed():
    """Install discovery metadata for spawned workers, using this interpreter."""
    import importlib.metadata
    from pathlib import Path
    import subprocess
    import sys

    entries = importlib.metadata.entry_points(group='vllm.general_plugins')
    if any(entry.value == 'qwenvl.vllm_audio:register' for entry in entries):
        return
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(root),
                    '--no-deps', '--no-build-isolation'], check=True)


def register():
    from vllm.model_executor.models import ModelRegistry
    for name in ("Qwen3_5AudioForConditionalGeneration", "Qwen3_5MoeAudioForConditionalGeneration"):
        ModelRegistry.register_model(name, f"qwenvl.vllm_audio.model:{name}")
    from .media import register_video_loader
    register_video_loader()
