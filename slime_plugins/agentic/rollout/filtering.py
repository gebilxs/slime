"""Group filters matching ``slime.rollout.fully_async_rollout`` semantics.

Standalone so login-node tests need not import SGLang. Length is ``len(tokens)``,
never character length of ``response``.
"""

from __future__ import annotations

import math


def _reward(sample, args=None):
    getter = getattr(sample, "get_reward_value", None)
    if callable(getter):
        try:
            return float(getter(args))
        except Exception:
            pass
    value = getattr(sample, "reward", None)
    if isinstance(value, dict):
        value = value.get("reward", value.get("score"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_zero_std_group(group, args=None, *, eps=1e-6):
    if len(group) < 2:
        return False
    rewards = [_reward(s, args) for s in group]
    if any(r is None for r in rewards):
        return False
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    return math.sqrt(var) <= eps


def is_overlong_group(group, max_tokens):
    if not max_tokens or max_tokens <= 0:
        return False
    return any(len(getattr(s, "tokens", None) or []) > max_tokens for s in group)


def filter_groups(
    groups,
    *,
    zero_std=True,
    max_response_tokens=None,
    max_tokens=None,
    reject_budget=None,
    args=None,
):
    cap = max_tokens if max_tokens is not None else max_response_tokens
    out = []
    rejected = 0
    for group in groups:
        if cap and is_overlong_group(group, int(cap)):
            continue
        if zero_std and is_zero_std_group(group, args) and (
            reject_budget is None or rejected < reject_budget
        ):
            rejected += 1
            continue
        out.append(group)
    return out


__all__ = ["filter_groups", "is_overlong_group", "is_zero_std_group"]
