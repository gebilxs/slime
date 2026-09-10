"""Pack ECHO world-loss fields from Samples into slime train_data.

No torch. Used by ray.rollout and unit-tested on CPU-only login nodes.
"""

from __future__ import annotations

from typing import Any

# Must appear in Megatron train get_batch keys or λ L_Env never reaches the GPU.
ENV_LOSS_BATCH_KEYS = ("world_loss_masks", "echo_full_obs_counts")


def pack_world_loss_fields(samples: list[Any]) -> dict[str, list] | None:
    """Return world_loss_masks + echo_full_obs_counts, or None if unused.

    Each echo_full_obs_counts[i] is |O| broadcast to response_length so packed
    CE can divide per token (paper full_observation_tokens).
    """
    if not any(getattr(s, "world_loss_mask", None) is not None for s in samples):
        return None
    world_loss_masks: list[list[int]] = []
    echo_full_obs_counts: list[list[float]] = []
    for sample in samples:
        rl = int(sample.response_length)
        wmask = getattr(sample, "world_loss_mask", None)
        if wmask is None:
            wmask = [0] * rl
        else:
            wmask = list(wmask)
            if len(wmask) != rl:
                raise ValueError(
                    f"world_loss_mask length {len(wmask)} != response_length {rl}"
                )
        if getattr(sample, "remove_sample", False):
            wmask = [0] * rl
        world_loss_masks.append(wmask)
        foc = getattr(sample, "echo_full_obs_count", None)
        if foc is None:
            foc = sum(wmask)
        echo_full_obs_counts.append([float(foc)] * rl)
    return {
        "world_loss_masks": world_loss_masks,
        "echo_full_obs_counts": echo_full_obs_counts,
    }
