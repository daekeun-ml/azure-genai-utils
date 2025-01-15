import importlib
from typing import Any
from .qa_generator import (
    QAType,
    QADataGenerator,
    CustomQADataGenerator,
    convert_to_oai_format,
    generate_qas,
)

_module_lookup = {
    "QAType": "qa_generator.qa_types",
    "QADataGenerator": "qa_generator.QADataGenerator",
    "CustomQADataGenerator": "qa_generator.CustomQADataGenerator",
    "convert_to_oai_format": "qa_generator.convert_to_oai_format",
    "generate_qas": "qa_generator.generate_qas",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "QAType",
    "QADataGenerator",
    "CustomQADataGenerator",
    "convert_to_oai_format",
    "generate_qas",
]
