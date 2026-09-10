"""Sync slime ``--rollout-function-path`` entry. Do not make this async."""

from __future__ import annotations


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Same contract as ``slime.rollout.fully_async_rollout.generate_rollout_fully_async``."""
    from slime.rollout.fully_async_rollout import generate_rollout_fully_async as _impl

    return _impl(args, rollout_id, data_buffer, evaluation=evaluation)


fully_async_rollout = generate_rollout_fully_async

__all__ = ["fully_async_rollout", "generate_rollout_fully_async"]
