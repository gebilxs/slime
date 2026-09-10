"""Shared types for the agentic plugin facade.

Training samples stay ``slime.utils.types.Sample``. Episode layout stays
``core.episode.EpisodeResult``. This module does not define a second Sample.
"""

from __future__ import annotations

from typing import Any

from core.episode import EpisodeResult
from core.episode import apply_episode_to_sample as _apply_core

Transcript = list[dict[str, str]]


def apply_episode_to_sample(
    sample,
    result: EpisodeResult,
    backend: str = "",
    *,
    backend_key: str = "env_backend",
    obs_count_attr: str | None = "echo_full_obs_count",
):
    """Adapt a protocol result onto slime's Sample. Leaves ``sample.reward`` unset."""
    return _apply_core(
        sample,
        result,
        backend,
        backend_key=backend_key,
        obs_count_attr=obs_count_attr,
    )


try:
    from slime.rollout.base_types import RolloutFnTrainOutput as RolloutOutput
except Exception:  # pragma: no cover - login node without slime internals
    class RolloutOutput:  # type: ignore[no-redef]
        def __init__(self, samples, metrics=None):
            self.samples = samples
            self.metrics = metrics or {}


__all__ = ["EpisodeResult", "RolloutOutput", "Transcript", "apply_episode_to_sample"]
