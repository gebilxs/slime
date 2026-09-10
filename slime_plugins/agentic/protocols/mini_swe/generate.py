"""mini-SWE generate: the primary training harness, plus the slime entry.

Turn structure mirrors mini-swe-agent's message list: the model generates one
assistant turn (trainable, with per-token logprobs), the turn is closed with
<|im_end|>, the env feedback — command observation or format-error text — is
appended as the NEXT USER MESSAGE (never inside the assistant turn), then a
fresh assistant turn is opened. finish_reason=length is upstream's
FormatError-with-length path: feedback, episode continues; a complete action
that made it in is still executed.

Shared machinery comes from ``core`` (transcript, result shape, budget
arithmetic, chat template, env pool). What lives here is what makes this
harness mini-SWE: the mswea prompt templates, and an env that owns action
parsing and submission detection.

``sample.reward`` is left unset. Verifier score goes in metadata so
``--custom-rm-path slime_plugins.agentic.protocols.mini_swe.reward.reward_func`` can score.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Awaitable, Callable

from core.chat_format import apply_chat_template, chat_markers, ends_with_im_end
from core.chat_format import sglang_stop_matched
from core.environ import getenv
from core.env_pool import _docker_sem, env_timed
from core.episode import (
    MIN_TURN_BUDGET,
    EpisodeResult,
    decode_turn,
    remaining_context,
    turn_sampling_params,
)
from core.episode import apply_episode_to_sample as _apply_to_sample
from core.tasks import instruction_for_task, resolve_task_path
from core.transcript import Transcript
from .mswea import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE

logger = logging.getLogger(__name__)

TurnGenerate = Callable[[list[int], dict], Awaitable[dict]]

__all__ = [
    "generate",
    "apply_episode_to_sample",
    "run_mini_swe_episode",
    "build_env",
    "EpisodeResult",
    "TurnGenerate",
]

# Default consecutive format errors tolerated before the episode is cut short.
_DEFAULT_FORMAT_CAP = 3


def build_env(sample, max_turns: int):
    backend = getenv("ENV_BACKEND", "mock")
    if backend == "mock":
        from .mock_env import MockSweEnv

        return MockSweEnv(sample.prompt, max_turns=max_turns), backend
    if backend == "docker":
        from .docker_env import DockerSweEnv

        task_path = resolve_task_path(sample)
        meta = dict(getattr(sample, "metadata", None) or {})
        image = meta.get("image") or getenv("DOCKER_IMAGE")
        if not image:
            from core.sandbox import resolve_docker_image

            image = resolve_docker_image(task_path)
        env = DockerSweEnv(
            instruction=instruction_for_task(sample, task_path),
            task_path=task_path,
            max_turns=max_turns,
            max_obs_chars=int(getenv("MAX_OBS_CHARS", "10000")),
            command_timeout=int(getenv("COMMAND_TIMEOUT", "120")),
            run_verifier=getenv("RUN_VERIFIER", "1") != "0",
            image=image or None,
            dataset=meta.get("dataset"),
        )
        return env, backend
    raise NotImplementedError(
        f"ENV_BACKEND={backend!r} unsupported for mini_swe; use mock|docker"
    )


def _initial_messages(task_text: str) -> list[dict]:
    """mini-swe-agent's two opening messages, verbatim templates."""
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": INSTANCE_TEMPLATE.format(task=task_text.strip())},
    ]


def _judge_transcript(response: str) -> str:
    """Trim the trajectory to what reward.py's llm-judge modes can carry."""
    if len(response) <= 48000:
        return response
    head, tail = response[:12000], response[-36000:]
    return f"{head}\n\n...[middle of trajectory truncated]...\n\n{tail}"


async def run_mini_swe_episode(
    *,
    prompt: str,
    env,
    tokenizer,
    generate_turn: TurnGenerate,
    sampling_params: dict | None = None,
    max_turns: int = 16,
    max_ctx: int = 16384,
    evaluation: bool = False,
    label: str = "",
) -> EpisodeResult:
    """One trajectory. Does not set a GRPO reward.

    ``prompt`` is accepted for caller symmetry; the task text comes from
    ``env.reset()``, which is authoritative.
    """
    del prompt, evaluation, label  # reserved for future metadata

    env_time = [0.0]
    task_text = await env_timed(env_time, env.reset)
    prompt_text = apply_chat_template(tokenizer, _initial_messages(task_text))
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    # Observations always ride in a user message for this harness.
    markers = chat_markers(prompt_text)

    tr = Transcript(tokenizer)
    final_reward = 0.0
    abort_reason = "unknown"
    status = "truncated"
    num_turns = 0
    turn_gen_lens: list[int] = []
    cache_cached_tokens = 0
    cache_prompt_tokens = 0
    n_format_errors = 0
    n_consec_format_errors = 0
    n_commands = 0
    submitted = False
    submission = ""

    format_cap = int(
        getenv("MAX_CONSECUTIVE_PARSE_FAIL", str(_DEFAULT_FORMAT_CAP))
        or _DEFAULT_FORMAT_CAP
    )
    per_turn_cap = int(getenv("MAX_NEW_TOKENS_PER_TURN", "0") or 0)
    episode_budget = float(getenv("EPISODE_BUDGET_S", "0") or 0)
    t_episode0 = time.monotonic()

    try:
        for _turn in range(max_turns):
            if episode_budget > 0 and (time.monotonic() - t_episode0) > episode_budget:
                status = "truncated"
                abort_reason = "time_budget"
                if getattr(env, "ever_started", True):
                    final_reward = await env_timed(env_time, env.force_done)
                break

            budget = remaining_context(max_ctx, len(prompt_ids), len(tr))
            if budget < MIN_TURN_BUDGET:
                status = "truncated"
                abort_reason = "context_budget"
                break

            output = await generate_turn(
                prompt_ids + tr.token_ids,
                turn_sampling_params(sampling_params, budget=budget, per_turn_cap=per_turn_cap),
            )
            meta = output.get("meta_info") or {}
            cache_cached_tokens += int(meta.get("cached_tokens") or 0)
            cache_prompt_tokens += int(meta.get("prompt_tokens") or 0)
            decoded = decode_turn(output, tokenizer)
            if decoded is None:
                status = "failed"
                abort_reason = "no_logprobs"
                break

            cur_ids, cur_logp, text = decoded
            matched = sglang_stop_matched(meta)
            if matched and not text.rstrip().endswith(matched):
                text = text + matched
            tr.add_generated(text, cur_ids, cur_logp)
            num_turns += 1
            turn_gen_lens.append(len(cur_ids))

            gen_truncated = (meta.get("finish_reason") or {}).get("type") == "length"

            # Close the assistant turn exactly like the template would. A
            # model-emitted <|im_end|> is already in cur_ids (trainable);
            # otherwise inject a non-trainable one.
            tr.add_template("\n" if ends_with_im_end(tokenizer, cur_ids) else "<|im_end|>\n")

            step = await env_timed(env_time, env.step, text, gen_truncated)
            final_reward = step.reward
            if step.command:
                n_commands += 1
            if step.is_format_error:
                n_format_errors += 1
                n_consec_format_errors += 1
            else:
                n_consec_format_errors = 0
            if step.submitted:
                submitted = True

            # The env feedback is the next user message; then reopen a fresh
            # assistant turn (mini-swe-agent's message structure, verbatim).
            tr.add_env_feedback(markers, ((step.observation, True),))

            if step.submitted:
                submission = (step.observation or "")[:500]

            if n_consec_format_errors >= format_cap > 0 and not step.done:
                final_reward = await env_timed(env_time, env.force_done)
                status = "truncated"
                abort_reason = "format_error"
                break

            if step.done:
                if step.max_turns_forced:
                    status = "truncated"
                    abort_reason = "max_turns"
                else:
                    status = "completed"
                    abort_reason = "done"
                break
        else:
            status = "truncated"
            abort_reason = "max_turns"
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                await env_timed(env_time, close)
            except Exception as exc:  # noqa: BLE001
                logger.warning("env.close failed: %s", exc)

    env_never_started = 0
    if getenv("ENV_BACKEND", "mock") == "docker":
        env_never_started = int(not bool(getattr(env, "ever_started", True)))

    metadata = {
        "verifier_score": float(final_reward),
        # judge inputs (truncated): reward.py's llm-judge modes read these
        "task_text": task_text.strip()[:6000],
        "transcript": _judge_transcript(tr.text),
        "abort_reason": abort_reason,
        "num_turns": num_turns,
        "max_turns": max_turns,
        "max_ctx": max_ctx,
        "prompt_len": len(prompt_ids),
        "response_len": len(tr),
        "turn_gen_lens": turn_gen_lens,
        "n_format_errors": n_format_errors,
        "n_commands": n_commands,
        "submitted": int(submitted),
        "submission": submission,
        "status": status,
        "env_time_s": round(env_time[0], 3),
        "episode_elapsed_s": round(time.monotonic() - t_episode0, 1),
        "env_never_started": env_never_started,
        "cache_cached_tokens": cache_cached_tokens,
        "cache_prompt_tokens": cache_prompt_tokens,
    }
    return EpisodeResult(
        status=status,
        prompt_ids=prompt_ids,
        response=tr.text,
        response_token_ids=tr.token_ids,
        loss_mask=tr.loss_mask,
        world_loss_mask=tr.world_loss_mask,
        rollout_log_probs=tr.log_probs,
        metadata=metadata,
        verifier_score=float(final_reward),
        full_obs_token_count=tr.obs_token_count,
    )


def apply_episode_to_sample(sample, result: EpisodeResult, backend: str) -> Any:
    """Write episode fields onto a slime Sample. Does not set sample.reward."""
    return _apply_to_sample(sample, result, backend, backend_key="mini_swe_env_backend")


async def generate(args: Any, sample: Any, sampling_params: dict, evaluation: bool = False):
    """slime --custom-generate-function-path entry (imports slime lazily)."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post

    assert not isinstance(sample.prompt, list), "mini_swe expects a string prompt"
    state = GenerateState(args)
    tokenizer = state.tokenizer
    max_turns = int(getenv("MAX_TURNS", getattr(args, "echo_max_turns", 16) or 16))
    env, backend = build_env(sample, max_turns)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    async def generate_turn(input_ids: list[int], sp: dict) -> dict:
        return await post(
            url,
            {"input_ids": input_ids, "sampling_params": sp, "return_logprob": True},
        )

    sem = _docker_sem() if backend == "docker" else None

    async def _run():
        result = await run_mini_swe_episode(
            prompt=sample.prompt,
            env=env,
            tokenizer=tokenizer,
            generate_turn=generate_turn,
            sampling_params=sampling_params,
            max_turns=max_turns,
            max_ctx=int(getenv("MAX_CONTEXT_LEN", "16384")),
            evaluation=evaluation,
            label=str(getattr(sample, "label", "") or ""),
        )
        return apply_episode_to_sample(sample, result, backend)

    if sem is None:
        return await _run()
    async with sem:
        return await _run()
