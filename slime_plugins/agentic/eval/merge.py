"""Merge evalkit-shaped dumps. Protocol / n / dataset keys must match."""

from __future__ import annotations


def merge_dumps(dumps):
    if not dumps:
        return {}
    first = dumps[0]
    out = dict(first)
    for other in dumps[1:]:
        for key in ("protocol", "n_samples_per_prompt"):
            if other.get(key) != first.get(key):
                raise ValueError(f"incompatible {key}")
        left = first.get("datasets") or {}
        right = other.get("datasets") or {}
        if left or right:
            merged = dict(left)
            for name, entry in right.items():
                if name in merged:
                    raise ValueError(f"duplicate dataset {name}")
                merged[name] = entry
            out["datasets"] = merged
            continue
        if set(other.get("task_ids", ())) & set(first.get("task_ids", ())):
            raise ValueError("duplicate task")
        out["task_ids"] = list(first.get("task_ids", [])) + list(other.get("task_ids", []))
        out["rewards"] = list(first.get("rewards", [])) + list(other.get("rewards", []))
    return out


__all__ = ["merge_dumps"]
