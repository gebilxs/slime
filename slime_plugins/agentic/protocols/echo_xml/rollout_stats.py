"""slime --custom-rollout-log-function-path hook.

One wandb row per rollout step: slime's default rollout/ + perf/ metrics plus
echo/* diagnostics parsed from ``sample.metadata`` (written by generate.py):
parse-fail rate, abort-reason mix, verifier scores, env wall time, and the
fraction of episodes that never needed a sandbox container (lazy create).

Returning True tells slime its default logging already ran here. On any
exception we return False so slime's default path still logs the step —
this hook must never kill a rollout actor.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from core.rollout_stats import (
    as_float,
    flatten_samples,
    meta,
    percentile,
    task_id_of,
    write_hit_records,
)

logger = logging.getLogger(__name__)

# Private aliases kept for callers that imported these before they moved.
_meta = meta
_as_float = as_float
_percentile = percentile


def sample_hit_record(sample: Any, rollout_id: int) -> dict[str, Any]:
    """One JSONL row: task identity + verifier/reward. No trajectory text."""
    meta = _meta(sample)
    reward = _as_float(getattr(sample, "reward", None))
    return {
        "rollout_id": int(rollout_id),
        "task_id": task_id_of(sample) or None,
        "label": str(getattr(sample, "label", "") or ""),
        "dataset": meta.get("dataset"),
        "verifier_score": float(meta.get("verifier_score") or 0.0),
        "reward": reward,
        "abort_reason": str(meta.get("abort_reason") or "unknown"),
        "status": str(meta.get("status") or getattr(sample, "status", "") or ""),
        "num_turns": int(meta.get("num_turns") or 0),
        "n_parse_fail": int(meta.get("n_parse_fail") or 0),
        "env_never_started": int(meta.get("env_never_started") or 0),
    }


def hits_path_from_args(args: Any) -> Path | None:
    save = getattr(args, "save", None)
    if not save:
        return None
    return Path(str(save)) / "echo_hits.jsonl"


def echo_metrics(samples: list) -> dict[str, float]:
    """Pure function over flat samples; unit-testable without slime."""
    n = len(samples)
    if n == 0:
        return {}

    aborts = Counter(str(_meta(s).get("abort_reason") or "unknown") for s in samples)
    parse_fail = sum(1 for s in samples if int(_meta(s).get("n_parse_fail") or 0) > 0)
    never_started = sum(1 for s in samples if _meta(s).get("env_never_started") == 1)
    gen_trunc = sum(1 for s in samples if int(_meta(s).get("n_gen_trunc") or 0) > 0)
    verifier = [float(_meta(s).get("verifier_score") or 0.0) for s in samples]
    env_times = [float(_meta(s).get("env_time_s") or 0.0) for s in samples]
    turns = [float(_meta(s).get("num_turns") or 0) for s in samples]
    done = sum(1 for s in samples if str(_meta(s).get("status")) == "completed")

    out: dict[str, float] = {
        "echo/parse_fail_rate": parse_fail / n,
        "echo/done_rate": done / n,
        "echo/verifier_mean": sum(verifier) / n,
        "echo/verifier_pos_rate": sum(1 for v in verifier if v > 0) / n,
        "echo/turns_mean": sum(turns) / n,
        "echo/env_never_started_rate": never_started / n,
        "echo/gen_trunc_rate": gen_trunc / n,
    }
    for reason, cnt in sorted(aborts.items()):
        out[f"echo/abort_{reason}"] = cnt / n
    # Per-dataset verifier split (mixed pools): without this, a hard/unfiltered
    # subset's zeros hide whether the learnable subset is improving.
    by_ds: dict[str, list[float]] = {}
    for s in samples:
        ds = str(_meta(s).get("dataset") or "unknown")
        by_ds.setdefault(ds, []).append(float(_meta(s).get("verifier_score") or 0.0))
    for ds, vals in sorted(by_ds.items()):
        if len(vals) >= 4:
            out[f"echo/verifier_mean_{ds}"] = sum(vals) / len(vals)
    if env_times and max(env_times) > 0:
        ordered = sorted(env_times)
        out |= {
            "echo/env_time_mean": sum(env_times) / n,
            "echo/env_time_p50": _percentile(ordered, 0.5),
            "echo/env_time_p90": _percentile(ordered, 0.9),
            "echo/env_time_max": ordered[-1],
        }
    return out


def log_rollout(rollout_id, args, samples, rollout_extra_metrics=None, rollout_time=0.0) -> bool:
    """slime custom rollout log hook. True = step fully logged here."""
    try:
        from slime.ray.rollout import (
            compute_metrics_from_samples,
            compute_perf_metrics_from_samples,
        )
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix

        samples = flatten_samples(samples)
        log_dict = {**(rollout_extra_metrics or {})}
        log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
        log_dict |= dict_add_prefix(
            compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/"
        )
        log_dict |= echo_metrics(samples)
        echo_brief = {k: round(v, 4) for k, v in log_dict.items() if k.startswith("echo/")}
        logger.info("echo rollout %s: %s", rollout_id, echo_brief)

        step = compute_rollout_step(args, rollout_id)
        log_dict["rollout/step"] = step
        logging_utils.log(args, log_dict, step_key="rollout/step")
        try:
            hits_path = hits_path_from_args(args)
            if hits_path is not None:
                write_hit_records(
                    hits_path,
                    [sample_hit_record(sample, rollout_id) for sample in samples],
                )
        except Exception:  # noqa: BLE001
            logger.exception("echo hit dump failed; continuing")
        return True
    except Exception:  # noqa: BLE001
        logger.exception("echo rollout_stats failed; falling back to slime default logging")
        return False
