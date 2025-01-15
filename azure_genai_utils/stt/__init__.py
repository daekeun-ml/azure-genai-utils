import importlib
from typing import Any
from .augment import get_audio_augments_baseline

_module_lookup = {
    "get_audio_augments_baseline": "stt.augment",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["get_audio_augments_baseline"]
