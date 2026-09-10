"""Protocol registry: mini_swe and echo_xml live in this package."""

from __future__ import annotations

from . import echo_xml, mini_swe

_PROTOCOLS = {mini_swe.name: mini_swe, echo_xml.name: echo_xml}


def get_protocol(name: str):
    key = str(name).lower().replace("-", "_")
    try:
        return _PROTOCOLS[key]
    except KeyError:
        raise ValueError(
            f"unknown agentic protocol {name!r}; choose from {sorted(_PROTOCOLS)}"
        ) from None


def register_protocol(name: str, module) -> None:
    if not name or not hasattr(module, "generate"):
        raise ValueError("protocol must expose generate")
    _PROTOCOLS[str(name).lower().replace("-", "_")] = module


def available_protocols() -> tuple[str, ...]:
    return tuple(sorted(_PROTOCOLS))


__all__ = [
    "available_protocols",
    "echo_xml",
    "get_protocol",
    "mini_swe",
    "register_protocol",
]
