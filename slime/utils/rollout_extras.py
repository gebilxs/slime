"""Resolve the optional rollout-extras provider (``--rollout-extras-path``).

Lets a plugin add fields to the rollout -> train hand-off without editing
core: extra rollout tensors and their dtypes, the extra keys the Megatron
train ``get_batch`` must request, and a per-step packing function.

Core only knows the attribute names below. When the flag is unset every
accessor returns an empty value, so the default path is unchanged.
"""

from __future__ import annotations

from typing import Any

from slime.utils.misc import load_function

_CACHE: dict[str | None, Any] = {}


def rollout_extras(args) -> Any | None:
    """Return the configured provider, or None. Resolved once per path."""
    path = getattr(args, "rollout_extras_path", None) if args is not None else None
    if not path:
        return None
    if path not in _CACHE:
        _CACHE[path] = load_function(path)
    return _CACHE[path]


def extra_tensor_dtypes(args) -> dict:
    provider = rollout_extras(args)
    return dict(getattr(provider, "TENSOR_DTYPES", {}) or {}) if provider else {}


def extra_batch_keys(args) -> tuple[str, ...]:
    """Keys the train ``get_batch`` must request for the plugin to work."""
    provider = rollout_extras(args)
    return tuple(getattr(provider, "BATCH_KEYS", ()) or ()) if provider else ()


def extra_passthrough_keys(args) -> tuple[str, ...]:
    """Extra keys to carry through the rollout-data broadcast."""
    provider = rollout_extras(args)
    if not provider:
        return ()
    return tuple(getattr(provider, "BATCH_KEYS", ()) or ()) + tuple(
        getattr(provider, "PASSTHROUGH_KEYS", ()) or ()
    )


def pack_extras(args, samples: list) -> dict | None:
    provider = rollout_extras(args)
    pack = getattr(provider, "pack", None) if provider else None
    return pack(samples) if pack is not None else None
