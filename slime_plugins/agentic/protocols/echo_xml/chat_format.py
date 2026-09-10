"""Back-compat shim. Moved to ``core.chat_format`` (harness-neutral)."""

from __future__ import annotations

from core.chat_format import (
    apply_chat_template,
    chat_markers,
    ends_with_im_end,
    sglang_stop_matched,
)

__all__ = ["apply_chat_template", "chat_markers", "ends_with_im_end", "sglang_stop_matched"]
