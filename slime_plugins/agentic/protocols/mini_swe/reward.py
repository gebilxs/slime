"""mini-SWE training reward: docker verifier + optional LLM judge.

Modes (REWARD_MODE, set from yaml protocol.reward_mode):
- ``oracle`` (default): pure binary verifier (0/1 from /oracle/tests/test.sh).
  v1 evidence: echo_xml's §8 lesson says additive format terms get hacked.
- ``oracle_judge``: terminal-bench-rl weighting — 0.65 * oracle + 0.35 * judge.
  Judge outage falls back to oracle-only (never poison training on API flakes).
- ``judge``: pure judge score; judge failure → 0.0 (reference behavior).

In ALL modes the oracle verifier still runs (run_verifier=1 in the yamls) and
``verifier_score`` keeps flowing into wandb via rollout_stats (mini/verifier_mean
*) — judge lanes are monitored against true oracle accuracy. Judge outcomes land
in metadata (judge_score / judge_error) for mini/judge_* metrics.
"""

from __future__ import annotations

import os
from typing import Any

from core.environ import getenv

# Weights from Danau5tin/terminal-bench-rl (calculate_reward.py): tests 65%,
# judge 35%. Judge scores HOW the agent worked; oracle verifies WHETHER it did.
ORACLE_WEIGHT = 0.65
JUDGE_WEIGHT = 0.35
# Per-format-error-turn multiplicative decay (judge modes only) — see bottom of
# reward_func for rationale. 0.85^n: 1 broken turn 0.85x, 3 turns 0.61x.
FORMAT_ERROR_DECAY = 0.85


def _meta(sample: Any) -> dict:
    md = getattr(sample, "metadata", None)
    return md if isinstance(md, dict) else {}


def _verifier(md: dict, sample: Any) -> float:
    verifier = md.get("verifier_score")
    if verifier is None:
        # generate still wrote sample.reward (0805 compat) — do not invent 0.
        existing = getattr(sample, "reward", None)
        if existing is not None and not isinstance(existing, dict):
            return float(existing)
        verifier = 0.0
    return float(verifier)


async def reward_func(args, sample, **kwargs) -> float:
    """slime --custom-rm-path entry. Signature matches slime docs."""
    md = _meta(sample)
    verifier = _verifier(md, sample)
    mode = getenv("REWARD_MODE", "oracle")
    if mode == "oracle":
        return verifier

    from .judge import judge_episode

    task_text = str(md.get("task_text") or getattr(sample, "prompt", "") or "")
    transcript = str(md.get("transcript") or "")
    score, err = await judge_episode(task_text, transcript)
    md["judge_error"] = int(score is None)
    if err:
        md["judge_error_detail"] = err
    if score is not None:
        md["judge_score"] = float(score)

    if mode == "judge":
        base = float(score) if score is not None else 0.0
    elif mode == "oracle_judge":
        base = verifier if score is None else ORACLE_WEIGHT * verifier + JUDGE_WEIGHT * score
    else:
        raise ValueError(f"unknown REWARD_MODE={mode!r}")

    # 2026-08-31 format gate (post-drift fix): turns rejected by the env as
    # format errors are pure waste; both judge channels tolerate them (judge
    # scored broken==clean 0.55 in pair tests), so penalize deterministically.
    # Multiplicative decay — bounded, monotone, unhackable (echo_xml §8:
    # additive format terms get hacked). 3 broken turns ≈ 0.61x.
    n_fmt = int(md.get("n_format_errors") or 0)
    if n_fmt > 0:
        md["format_gate_factor"] = FORMAT_ERROR_DECAY ** n_fmt
        base *= md["format_gate_factor"]
    return base
