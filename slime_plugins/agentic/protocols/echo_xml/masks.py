"""ECHO observation splitting, plus the shared token lanes re-exported.

``append_tokens`` / ``world_flags`` moved to ``core.tokens`` (they are
harness-neutral); ``split_obs`` stays here because it parses the XML
protocol's <command_output> span.
"""

from __future__ import annotations

import re

from core.tokens import append_tokens, world_flags

__all__ = ["append_tokens", "world_flags", "split_obs"]

_TOOL_RE = re.compile(r"<command_output>(.*?)</command_output>", re.DOTALL)


def split_obs(observation: str) -> tuple[str, str]:
    """Split observation into (warning_prefix, command_output_span)."""
    m = _TOOL_RE.search(observation or "")
    if not m:
        return "", observation or ""
    return observation[: m.start()], observation[m.start() : m.end()]
