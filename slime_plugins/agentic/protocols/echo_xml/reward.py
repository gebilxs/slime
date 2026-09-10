"""ECHO training reward: verifier / judge blend as --custom-rm-path.

generate() must leave sample.reward unset and write signals into metadata:

    verifier_score, judge_score (optional), n_parse_fail

Modes (REWARD_MODE, default oracle):
    oracle      — docker tests/test.sh 0/1
    blend       — (1-w)*verifier + w*judge
    judge_only  — judge only (0 if missing)

Shaping (ECHO_OVERLONG_SOFT_PUNISH, default 0 = off):
    Explicit switch for DAPO Soft Overlong Punishment (arXiv:2503.14476
    Eq.13) with the turn axis as length: turns <= max_turns - L_cache is
    penalty-free, then a linear ramp down to -ECHO_OVERLONG_PUNISH_SCALE at
    the turn cap. Episodes force-stopped at max_turns land on the full
    penalty. Switch on requires ECHO_OVERLONG_L_CACHE > 0 (fail-loud).

Shaping (ECHO_OVERLONG_TOKEN_SOFT_PUNISH, default 0 = off):
    Same DAPO soft punishment but on the generated-token axis: the length
    variable is the episode's total generated tokens (sum of per-turn
    completion lengths, observations excluded), the budget reference is
    ECHO_OVERLONG_TOKEN_L_MAX (default 8192), the free zone ends at
    L_MAX - ECHO_OVERLONG_TOKEN_L_CACHE, and the ramp bottoms at
    -ECHO_OVERLONG_TOKEN_PUNISH_SCALE (default 0.5). Unlike the turn axis
    this never taxes interaction depth: many lean turns pay nothing.
    Switch on requires ECHO_OVERLONG_TOKEN_L_CACHE > 0 (fail-loud).

This module does not call Harbor and does not talk to SGLang. Judge HTTP, if
enabled, is invoked here (not in generate).
"""

from __future__ import annotations

import os
from typing import Any

from core.environ import getenv


def reward_mode() -> str:
    explicit = getenv("REWARD_MODE") or os.environ.get("ECHO_JUDGE_MODE")
    if explicit:
        return explicit.strip().lower()
    if os.environ.get("ECHO_JUDGE_ENABLE", "0") == "1":
        return "blend"
    return "oracle"


def judge_weight() -> float:
    return float(os.environ.get("ECHO_JUDGE_WEIGHT", "0.35"))


def parse_fail_penalty() -> float:
    """Multiply final reward by this for each parse fail (default 0 = off)."""
    return float(os.environ.get("ECHO_PARSE_FAIL_PENALTY", "0"))


def format_coeff() -> float:
    """Additive format reward weight (default 0 = off). See RL_INFRA.md §8."""
    return float(os.environ.get("ECHO_FORMAT_COEFF", "0"))


def reward_scale() -> float:
    """Final affine scale on the scalar reward (default 1.0 = off).

    NOTE: slime GRPO group normalization (rollout.py _post_process_rewards)
    cancels ANY affine transform r -> scale*r + bias exactly (both group-mean
    subtraction and group-std division). Kept so the reference recipe's
    reward scaling/bias is discoverable in the train yaml; it is a no-op
    under group-normed GRPO with KL=0.
    """
    return float(getenv("REWARD_SCALE", "1.0"))


def reward_bias() -> float:
    """Final affine bias on the scalar reward (default 0.0 = off).

    Same no-op caveat as reward_scale() under GRPO group normalization.
    """
    return float(getenv("REWARD_BIAS", "0.0"))


def format_score(n_commands: int, num_turns: int, clean_done: bool) -> float:
    """Fraction of turns that emitted a valid action.

    Guards (anti-degenerate):
    - no command at all -> 0 (an instant <action>done</action> farms no credit);
    - a clean final done counts as one valid turn, a forced max-turns cut does not.
    """
    if num_turns <= 0 or n_commands <= 0:
        return 0.0
    valid = int(n_commands) + (1 if clean_done else 0)
    return min(1.0, valid / num_turns)


def overlong_enabled() -> bool:
    """Explicit on/off switch (yaml protocol.overlong_soft_punish)."""
    raw = os.environ.get("ECHO_OVERLONG_SOFT_PUNISH", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def overlong_l_cache() -> int:
    """Soft-zone width L_cache in turns (parameter only; the on/off switch
    is overlong_enabled())."""
    return int(os.environ.get("ECHO_OVERLONG_L_CACHE", "0"))


def overlong_punish_scale() -> float:
    """Penalty at the turn cap (default 1.0 = DAPO Eq.13)."""
    return float(os.environ.get("ECHO_OVERLONG_PUNISH_SCALE", "1.0"))


def overlong_turn_penalty(num_turns: int, max_turns: int, l_cache: int, scale: float = 1.0) -> float:
    """DAPO Soft Overlong Punishment (Eq.13), turns as the length axis.

    Non-positive: 0 while num_turns <= max_turns - l_cache, then linear in
    the soft zone down to -scale at the cap. Episodes force-stopped at
    max_turns (abort=max_turns) report num_turns == max_turns and take the
    full -scale. Penalty is ADDED to the base reward, so a success exactly
    at the cap nets verifier - scale.
    """
    if l_cache <= 0 or max_turns <= 0 or scale == 0.0:
        return 0.0
    free = max_turns - l_cache
    if num_turns <= free:
        return 0.0
    return -float(scale) * min(1.0, (num_turns - free) / float(l_cache))


def overlong_token_enabled() -> bool:
    """Explicit on/off switch (yaml protocol.overlong_token_soft_punish)."""
    raw = os.environ.get("ECHO_OVERLONG_TOKEN_SOFT_PUNISH", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def overlong_token_l_max() -> int:
    """Budget reference L_max in episode-total generated tokens."""
    return int(os.environ.get("ECHO_OVERLONG_TOKEN_L_MAX", "8192"))


def overlong_token_l_cache() -> int:
    """Soft-zone width L_cache in generated tokens (parameter only; the
    on/off switch is overlong_token_enabled())."""
    return int(os.environ.get("ECHO_OVERLONG_TOKEN_L_CACHE", "0"))


def overlong_token_punish_scale() -> float:
    """Penalty at the token budget (default 0.5 — turn-axis overdose lesson:
    scale=1.0 against a 0/1 verifier zeroes slow-but-correct episodes)."""
    return float(os.environ.get("ECHO_OVERLONG_TOKEN_PUNISH_SCALE", "0.5"))


def overlong_token_penalty(gen_tokens: int, l_max: int, l_cache: int, scale: float = 0.5) -> float:
    """DAPO Soft Overlong Punishment (Eq.13) on the generated-token axis.

    gen_tokens is the episode's total generated tokens (sum of per-turn
    completion lengths, observations excluded) — the honest DAPO analog of
    response length. Non-positive: 0 while gen_tokens <= l_max - l_cache,
    then linear in the soft zone down to -scale at l_max. Unlike the turn
    axis this never taxes interaction depth: a 14-turn episode of lean
    generations pays nothing, a 2-turn rumination blowout pays in full.
    """
    if l_cache <= 0 or l_max <= 0 or scale == 0.0:
        return 0.0
    free = l_max - l_cache
    if gen_tokens <= free:
        return 0.0
    return -float(scale) * min(1.0, (gen_tokens - free) / float(l_cache))


def score_from_signals(
    *,
    verifier: float,
    judge: float | None = None,
    mode: str | None = None,
    weight: float | None = None,
    parse_fail: int = 0,
    fallback: str | None = None,
    n_commands: int = 0,
    num_turns: int = 0,
    clean_done: bool = False,
    format_w: float | None = None,
    scale: float | None = None,
    bias: float | None = None,
    max_turns: int = 0,
    gen_tokens: int = 0,
) -> float:
    """Pure scoring. Unit-test this; slime only sees reward_func."""
    mode = (mode or reward_mode()).strip().lower()
    weight = judge_weight() if weight is None else float(weight)
    fallback = fallback or os.environ.get("JUDGE_FALLBACK", "verifier")
    verifier = float(verifier)

    if mode in ("oracle", "verifier", "off", ""):
        r = verifier
    elif mode in ("judge_only", "judge-only", "only"):
        r = 0.0 if judge is None else float(judge)
    elif mode == "blend":
        if judge is None:
            r = 0.0 if fallback == "zero" else verifier
        else:
            r = (1.0 - weight) * verifier + weight * float(judge)
    else:
        raise ValueError(f"unknown REWARD_MODE={mode!r}")

    penalty = parse_fail_penalty()
    if penalty > 0 and parse_fail > 0:
        r = r * max(0.0, 1.0 - penalty * int(parse_fail))

    fw = format_coeff() if format_w is None else float(format_w)
    if fw > 0:
        r = r + fw * format_score(n_commands, num_turns, clean_done)
    oc = overlong_l_cache()
    if overlong_enabled():
        if oc <= 0:
            raise ValueError(
                "ECHO_OVERLONG_SOFT_PUNISH=1 requires ECHO_OVERLONG_L_CACHE > 0 "
                "(set protocol.overlong_l_cache in the train yaml)"
            )
        r = r + overlong_turn_penalty(num_turns, max_turns, oc, overlong_punish_scale())
    if overlong_token_enabled():
        otc = overlong_token_l_cache()
        if otc <= 0:
            raise ValueError(
                "ECHO_OVERLONG_TOKEN_SOFT_PUNISH=1 requires ECHO_OVERLONG_TOKEN_L_CACHE > 0 "
                "(set protocol.overlong_token_l_cache in the train yaml)"
            )
        r = r + overlong_token_penalty(
            gen_tokens, overlong_token_l_max(), otc, overlong_token_punish_scale()
        )
    s = reward_scale() if scale is None else float(scale)
    b = reward_bias() if bias is None else float(bias)
    if s != 1.0 or b != 0.0:
        r = s * r + b
    return float(r)


def _meta(sample: Any) -> dict:
    md = getattr(sample, "metadata", None)
    return md if isinstance(md, dict) else {}


async def reward_func(args, sample, **kwargs) -> float:
    """slime --custom-rm-path entry. Signature matches slime docs.

    Judge HTTP runs here, never inside generate().
    """
    md = _meta(sample)
    verifier = md.get("verifier_score")
    if verifier is None:
        # generate still wrote sample.reward (0805 compat) — do not invent 0.
        existing = getattr(sample, "reward", None)
        if existing is not None and not isinstance(existing, dict):
            return float(existing)
        verifier = 0.0
    judge = md.get("judge_score")
    if judge is not None:
        judge = float(judge)
    mode = reward_mode()
    if judge is None:
        from .judge import needs_judge_http, score_trajectory

        if needs_judge_http(mode):
            judge = await score_trajectory(
                instruction=str(getattr(sample, "prompt", "") or md.get("instruction") or ""),
                trajectory=str(getattr(sample, "response", "") or md.get("trajectory") or ""),
                task_id=str(getattr(sample, "label", "") or md.get("task_id") or ""),
            )
            if judge is not None and isinstance(getattr(sample, "metadata", None), dict):
                sample.metadata["judge_score"] = float(judge)
    turn_lens = md.get("turn_gen_lens") or []
    gen_tokens = (
        sum(int(x) for x in turn_lens) if isinstance(turn_lens, (list, tuple)) else 0
    )
    return score_from_signals(
        verifier=float(verifier),
        judge=judge,
        parse_fail=int(md.get("n_parse_fail") or 0),
        n_commands=int(md.get("n_commands") or 0),
        num_turns=int(md.get("num_turns") or 0),
        clean_done=str(md.get("abort_reason")) == "done",
        max_turns=int(md.get("max_turns") or 0),
        gen_tokens=gen_tokens,
    )
