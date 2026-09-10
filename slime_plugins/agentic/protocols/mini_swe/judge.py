"""LLM-as-a-judge backend for the mini-SWE lane (terminal-bench-rl pattern).

Judge = k3 on the internal OpenAI-compatible gateway. One judge call per
episode at reward time: system = judge_prompt_miniswe.md, user = task
instruction + truncated trajectory transcript. Returns a 0..1 score parsed
from the judge's `score: X.XX` line; failures return None (caller decides the
fallback) and are always recorded in metadata for wandb.

Env overrides: JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL /
JUDGE_MAX_CHARS / JUDGE_TIMEOUT_S / JUDGE_CONCURRENCY.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from core.environ import getenv

# NOTE: `or <default>` not .get(..., default) — train.sh propagates
# JUDGE_API_KEY / JUDGE_MODEL as EMPTY strings when the job yaml
# leaves them unset, and .get only falls back when the var is absent.
_BASE_URL = getenv("JUDGE_BASE_URL") or "http://10.12.111.133:49183/v1"
_API_KEY = (
    getenv("JUDGE_API_KEY")
    or "sk-f189140cc365af129e17d7748c1a1133edc1a54063dc02fa1b1498c6f3752de8"
)
_MODEL = getenv("JUDGE_MODEL") or "k3"
_MAX_CHARS = int(getenv("JUDGE_MAX_CHARS", "48000"))
_TIMEOUT = float(getenv("JUDGE_TIMEOUT_S", "180"))
_CONCURRENCY = int(getenv("JUDGE_CONCURRENCY", "16"))

_PROMPT = (Path(__file__).parent / "judge_prompt_miniswe.md").read_text(encoding="utf-8")
_SCORE_RE = re.compile(r"score:\s*([01](?:\.\d{1,2})?)")

_sem: asyncio.Semaphore | None = None
_client = None


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(_CONCURRENCY)
    return _sem


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(base_url=_BASE_URL, api_key=_API_KEY, timeout=_TIMEOUT)
    return _client


def truncate_transcript(text: str, max_chars: int = _MAX_CHARS) -> str:
    """Keep the head (task setup, first actions) and the tail (verification,
    submission) — the middle of long trajectories is the least informative."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    tail = max_chars - head
    return text[:head] + "\n\n...[middle of trajectory truncated]...\n\n" + text[-tail:]


def parse_score(text: str) -> float | None:
    m = _SCORE_RE.search(text or "")
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(1))))


async def judge_episode(task_text: str, transcript: str) -> tuple[float | None, str | None]:
    """Return (score, error). score None ⇏ 0.0 — the caller picks the fallback."""
    user = (
        "Task instruction given to the agent:\n"
        + task_text.strip()[:6000]
        + "\n\nAgent trajectory (assistant turns and environment observations):\n"
        + truncate_transcript(transcript)
    )
    last_err: str | None = None
    async with _get_sem():
        for attempt in range(3):
            try:
                r = await _get_client().chat.completions.create(
                    model=_MODEL,
                    messages=[
                        {"role": "system", "content": _PROMPT},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                score = parse_score(r.choices[0].message.content or "")
                if score is not None:
                    return score, None
                last_err = "unparseable_score"
            except Exception as exc:  # noqa: BLE001 — judge must never kill a rollout
                last_err = f"{type(exc).__name__}: {exc}"[:200]
            await asyncio.sleep(min(2 ** attempt * 5, 30))
    return None, last_err
