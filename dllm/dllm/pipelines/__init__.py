import importlib

_pipeline_names = ["a2d", "bert", "dream", "editflow", "fastdllm", "llada", "llada2", "llada21", "rl"]

for _name in _pipeline_names:
    try:
        globals()[_name] = importlib.import_module(f".{_name}", package=__name__)
    except Exception:
        globals()[_name] = None

__all__ = _pipeline_names
