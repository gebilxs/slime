"""LLM-as-a-Judge for ECHO trajectories. Called from reward.py, not generate.

Uses slime_plugins.agentic.clients.openai_http. Default OFF: oracle reward never hits HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

from slime_plugins.agentic.clients.openai_http import OpenAIHttpClient

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = Path(__file__).parent / "judge_prompts" / "echo_terminal_judge_v1.md"
_prompt_cache: str | None = None


def _load_prompt() -> str:
    global _prompt_cache
    if _prompt_cache is None:
        path = Path(os.environ.get("JUDGE_PROMPT", str(_DEFAULT_PROMPT)))
        _prompt_cache = path.read_text(encoding="utf-8")
    return _prompt_cache


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n...[trajectory truncated]...\n" + text[-tail:]


def parse_judge_score(response_text: str) -> float | None:
    """Extract a 0..1 score. YAML fence, then looser regex."""
    text = (response_text or "").strip()
    block = text
    if "```yaml" in text:
        start = text.find("```yaml") + 7
        end = text.find("```", start)
        if end > start:
            block = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            block = text[start:end].strip()

    match = re.search(r"^score\s*:\s*([0-9.]+)\s*$", block, re.IGNORECASE | re.MULTILINE)
    if not match:
        for pattern in (
            r"score\s*[:=]\s*([0-9.]+)",
            r'"score"\s*:\s*([0-9.]+)',
            r"\bscore\b[^0-9]{0,12}(0?\.\d+|1\.0|0|1)(?![0-9])",
        ):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                break
    if not match:
        return None
    try:
        score = float(match.group(1))
    except ValueError:
        return None
    return score if 0.0 <= score <= 1.0 else None


def needs_judge_http(mode: str) -> bool:
    return mode in ("blend", "judge_only", "judge-only", "only")


async def score_trajectory(
    *,
    instruction: str,
    trajectory: str,
    task_id: str = "",
    client: OpenAIHttpClient | None = None,
) -> float | None:
    """Score one finished trajectory in [0, 1]. None = judge unavailable."""
    client = client or OpenAIHttpClient.from_env()
    if client is None:
        return None

    max_chars = int(os.environ.get("JUDGE_MAX_CHARS", "24000"))
    user_message = (
        f"# Task instruction\n{truncate_middle(instruction, 4000)}\n\n"
        f"# Agent trajectory\n```\n{truncate_middle(trajectory, max_chars)}\n```"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _load_prompt()},
        {"role": "user", "content": user_message},
    ]
    max_retries = int(os.environ.get("JUDGE_MAX_RETRIES", "3"))
    last_err = ""
    for attempt in range(max_retries):
        try:
            content = await asyncio.to_thread(client.chat, messages, temperature=0.0)
            score = parse_judge_score(content)
            if score is not None:
                return score
            last_err = f"unparseable response: {content[-200:]!r}"
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "Reply with exactly one line: `score: X.XX` "
                    "(a number between 0.00 and 1.00). No other text.",
                }
            )
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries - 1:
            await asyncio.sleep(min(2**attempt, 8))
    logger.warning("judge failed for %s after %d attempts: %s", task_id, max_retries, last_err)
    return None
