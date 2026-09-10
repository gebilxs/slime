"""Re-export the live evalkit pass@k. Signature is ``pass_at_k(n, c, k)``."""

from evalkit.passk import (
    DEFAULT_KS,
    DEFAULT_THRESHOLD,
    analyze_dump,
    analyze_flat,
    analyze_groups,
    group_flat,
    pass_at_k,
    passes_in_group,
)

__all__ = [
    "DEFAULT_KS",
    "DEFAULT_THRESHOLD",
    "analyze_dump",
    "analyze_flat",
    "analyze_groups",
    "group_flat",
    "pass_at_k",
    "passes_in_group",
]
