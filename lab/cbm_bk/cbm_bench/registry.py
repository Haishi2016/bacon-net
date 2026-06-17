"""Simple name-based registries for datasets and models.

This keeps datasets and models decoupled: a model never needs to know which
dataset it runs on, and the training driver only talks to the registries.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

# name -> factory(callable). Factories receive a config object/dict and return
# the constructed object (a dataset bundle or an nn.Module).
_DATASETS: Dict[str, Callable[..., Any]] = {}
_MODELS: Dict[str, Callable[..., Any]] = {}


def register_dataset(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a dataset factory under ``name``."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = name.lower()
        if key in _DATASETS:
            raise KeyError(f"Dataset '{name}' is already registered.")
        _DATASETS[key] = fn
        return fn

    return _wrap


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a model factory under ``name``."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = name.lower()
        if key in _MODELS:
            raise KeyError(f"Model '{name}' is already registered.")
        _MODELS[key] = fn
        return fn

    return _wrap


def build_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
    key = name.lower()
    if key not in _DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {sorted(_DATASETS)}"
        )
    return _DATASETS[key](*args, **kwargs)


def build_model(name: str, *args: Any, **kwargs: Any) -> Any:
    key = name.lower()
    if key not in _MODELS:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(_MODELS)}"
        )
    return _MODELS[key](*args, **kwargs)


def list_datasets() -> list[str]:
    return sorted(_DATASETS)


def list_models() -> list[str]:
    return sorted(_MODELS)
