import importlib
from typing import Any
from .bing import BingSearch

_module_lookup = {
    "BingSearch": "tools.bing",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["BingSearch"]
