"""Eval dump facade. Prefer ``evalkit.dump.dump_eval`` for slime hooks."""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.dump import dump_eval
from evalkit.passk import analyze_flat


def write_dump(path, *, protocol, dataset, n_samples_per_prompt, task_ids, rewards, passk=None):
    """Write one-dataset dump in the evalkit shape (``datasets`` + pass@k)."""
    n = int(n_samples_per_prompt)
    stats = analyze_flat(list(rewards), n) if rewards else {"passk": {}, "mean": 0.0}
    obj = {
        "protocol": protocol,
        "n_samples_per_prompt": n,
        "datasets": {
            dataset: {
                "n": n,
                "task_ids": list(task_ids),
                "rewards": list(rewards),
                "passk": passk or stats.get("passk") or {},
            }
        },
    }
    Path(path).write_text(json.dumps(obj), encoding="utf-8")
    return obj


__all__ = ["dump_eval", "write_dump"]
