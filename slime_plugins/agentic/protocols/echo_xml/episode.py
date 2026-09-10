"""ECHO XML episode: the reference turn loop and its flag surface.

``run_echo_episode`` drives one trajectory: generate a turn, parse the XML
action, step the sandbox, append the observation, repeat. It sets no GRPO
reward — the verifier score rides in metadata so
``slime_plugins.agentic.protocols.echo_xml.reward`` owns the scalar.

Shared machinery lives in ``core``: ``Transcript`` holds the token lanes,
``core.episode`` the result shape and budget arithmetic. What stays here is
what makes this harness the XML protocol: the ``<command>`` action space,
warning/observation splitting, and the turn-level GRPO bookkeeping.

``EpisodeFlags`` is re-read from the environment on every call rather than
cached at import: rollout workers are long-lived and the tests flip ECHO_*
between episodes.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

from core.chat_format import (
    apply_chat_template,
    chat_markers,
    ends_with_im_end,
    sglang_stop_matched,
)
from core.environ import getenv
from core.env_pool import env_timed
from core.episode import (
    MIN_TURN_BUDGET,
    EpisodeResult,
    decode_turn,
    remaining_context,
    turn_sampling_params,
)
from core.tokens import world_flags
from core.transcript import Transcript
from .masks import split_obs
from .xml import parse_action

logger = logging.getLogger(__name__)

__all__ = ["EpisodeFlags", "EpisodeStats", "EpisodeResult", "Transcript", "run_echo_episode"]

# RL_INFRA.md §12 follow-up: interrupt the in-context repetition trap. When the
# same command is resubmitted (or a command silently produces no output), the
# next user message carries a teaching hint instead of another dead observation.
_HINT_REPEAT = (
    "- You just executed a command identical to an earlier one in this episode; "
    "rerunning it will produce the same result. Change approach: run a small "
    "diagnostic first (e.g. ls the target path, echo $? for the exit code), "
    "then fix the command.\n"
)
_HINT_EMPTY_OBS = (
    "- Your last command produced no output; it may have failed silently. "
    "Check the exit code with `echo $?` or inspect the path with `ls`.\n"
)
_EMPTY_OBS_RE = re.compile(r"<command_output>\s*</command_output>")

_ACTION_STOPS = ("</command>", "</action>")
_DONE_ACTION = "<action>done</action>"


def _env_flag(name: str, default: str = "0") -> bool:
    return getenv(name, default) == "1"


def _env_number(name: str) -> str:
    return getenv(name, "0") or "0"


@dataclass(frozen=True)
class EpisodeFlags:
    """ECHO_* tunables for one episode. Built per call, never cached."""

    apply_chat_template: bool = True
    user_role_obs: bool = False
    gen_length_continue: bool = False
    stop_on_action: bool = False
    repeat_hint: bool = False
    max_parse_fail: int = 0
    per_turn_cap: int = 0
    episode_budget: float = 0.0

    @classmethod
    def from_env(cls) -> EpisodeFlags:
        return cls(
            apply_chat_template=_env_flag("APPLY_CHAT_TEMPLATE", "1"),
            # C1 (RL_INFRA.md §10/§11): observations as user-role messages with
            # proper template turn boundaries, not raw spans inside the
            # assistant turn.
            user_role_obs=_env_flag("USER_ROLE_OBS"),
            # Official-harness semantics: a turn that hits the per-turn cap is
            # clipped and the episode continues instead of dying on gen_length.
            gen_length_continue=_env_flag("ECHO_GEN_LENGTH_CONTINUE"),
            stop_on_action=_env_flag("STOP_ON_ACTION"),
            # Repetition / empty-output teaching hints (RL_INFRA.md §12).
            repeat_hint=_env_flag("ECHO_REPEAT_HINT"),
            max_parse_fail=int(_env_number("MAX_CONSECUTIVE_PARSE_FAIL")),
            per_turn_cap=int(_env_number("MAX_NEW_TOKENS_PER_TURN")),
            # RL_INFRA.md §11 speed pack: hard wall-clock ceiling per episode.
            # The env chain (create + turns*exec + verify) is the step-time
            # floor; stragglers gate the whole batch, so cap them and keep
            # partial credit.
            episode_budget=float(_env_number("EPISODE_BUDGET_S")),
        )

    def turn_sampling_params(self, base: dict | None, budget: int) -> dict:
        """Per-turn sampling params clamped to the remaining context budget."""
        return turn_sampling_params(
            base,
            budget=budget,
            per_turn_cap=self.per_turn_cap,
            stop=_ACTION_STOPS if self.stop_on_action else None,
        )


@dataclass
class EpisodeStats:
    """Per-turn counters reported through sample metadata."""

    num_turns: int = 0
    turn_gen_lens: list[int] = field(default_factory=list)
    # turn_token_spans[t] = [start, end) in response-token space covering the
    # turn's generated tokens + template seps + observation spans (trainable
    # positions inside are exactly loss_mask==1). turn_signals[t] records
    # whether the turn executed a command and whether it ended the episode.
    # Metadata-only; no effect on training unless TURN_ADV=1.
    turn_token_spans: list[list[int]] = field(default_factory=list)
    turn_signals: list[dict] = field(default_factory=list)
    n_warnings: int = 0
    n_command_outputs: int = 0
    # RL_INFRA.md §8 fix: count commands the env actually parsed and executed,
    # not raw "<command>" tags (think-block drafts hacked the format reward).
    n_commands: int = 0
    n_gen_trunc: int = 0
    # Prefix-cache observability: the router runs cache_aware, so turn N should
    # hit turn 1..N-1's KV; without these counts
    # rollout/prefix_cache_hit_rate silently reports 0.
    cache_cached_tokens: int = 0
    cache_prompt_tokens: int = 0

    def close_turn(self, span_start: int, span_end: int, *, executed: bool, done: bool) -> None:
        self.turn_token_spans.append([span_start, span_end])
        self.turn_signals.append({"executed": executed, "done": done})


def _never_started(env) -> int:
    """1 = a docker/harbor episode ended without ever starting a container."""
    if getenv("ENV_BACKEND", "mock") not in ("docker", "harbor"):
        return 0
    # ever_started survives close(), unlike started.
    return int(not bool(getattr(env, "ever_started", True)))


async def run_echo_episode(
    *,
    prompt: str,
    env,
    tokenizer,
    generate_turn,
    sampling_params: dict | None = None,
    max_turns: int = 8,
    max_ctx: int = 4096,
    world_target: str = "env_only",
    evaluation: bool = False,
    label: str = "",
) -> EpisodeResult:
    """One trajectory. Does not set a GRPO reward.

    ``prompt`` is accepted for caller symmetry but the first user message comes
    from ``env.reset()``, which is authoritative for sandbox tasks.
    """
    del prompt, evaluation, label  # reserved for future judge metadata

    env_time = [0.0]
    prompt_text = await env_timed(env_time, env.reset)
    flags = EpisodeFlags.from_env()
    if flags.apply_chat_template:
        prompt_text = apply_chat_template(tokenizer, [{"role": "user", "content": prompt_text}])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    markers = chat_markers(prompt_text) if flags.user_role_obs else None

    tr = Transcript(tokenizer)
    stats = EpisodeStats()
    final_reward = 0.0
    abort_reason = "unknown"
    status = "truncated"
    n_consec_parse_fail = 0
    ep_cmds: list[str] = []
    t_episode0 = time.monotonic()

    try:
        for _turn in range(max_turns):
            if flags.episode_budget > 0 and (
                time.monotonic() - t_episode0
            ) > flags.episode_budget:
                status = "truncated"
                abort_reason = "time_budget"
                if getattr(env, "ever_started", True):
                    # verify-only exit: keep partial credit, never count as done
                    step = await env_timed(env_time, env.step, _DONE_ACTION)
                    final_reward = step.reward
                break

            budget = remaining_context(max_ctx, len(prompt_ids), len(tr))
            if budget < MIN_TURN_BUDGET:
                status = "truncated"
                abort_reason = "context_budget"
                break

            output = await generate_turn(
                prompt_ids + tr.token_ids, flags.turn_sampling_params(sampling_params, budget)
            )
            span_start = len(tr)
            meta = output.get("meta_info") or {}
            stats.cache_cached_tokens += int(meta.get("cached_tokens") or 0)
            stats.cache_prompt_tokens += int(meta.get("prompt_tokens") or 0)
            decoded = decode_turn(output, tokenizer)
            if decoded is None:
                status = "failed"
                abort_reason = "no_logprobs"
                break

            cur_ids, cur_logp, text = decoded
            matched = sglang_stop_matched(meta)
            if matched in _ACTION_STOPS and not text.rstrip().endswith(matched):
                text = text + matched
            tr.add_generated(text, cur_ids, cur_logp)
            stats.num_turns += 1
            stats.turn_gen_lens.append(len(cur_ids))

            finish = (meta.get("finish_reason") or {}).get("type")
            if finish == "length":
                if not flags.gen_length_continue:
                    stats.close_turn(span_start, len(tr), executed=False, done=False)
                    status = "truncated"
                    abort_reason = "gen_length"
                    break
                # Clipped turn keeps going — fall through to parse/env with the
                # truncated text; a complete <command>...</command> inside
                # still gets executed.
                stats.n_gen_trunc += 1

            if markers is not None:
                # C1: close the assistant turn exactly like the template would.
                # A model-emitted <|im_end|> is already in cur_ids (trainable);
                # otherwise inject a non-trainable one (stop-string / no-EOS path).
                tr.add_template("\n" if ends_with_im_end(tokenizer, cur_ids) else "<|im_end|>\n")

            cmd, _sig_done, _sig_warn = parse_action(text)
            hint_parts: list[str] = []
            if cmd:
                stats.n_commands += 1
                if flags.repeat_hint and cmd in ep_cmds:
                    hint_parts.append(_HINT_REPEAT)
                ep_cmds.append(cmd)

            step = await env_timed(env_time, env.step, text)
            final_reward = step.reward
            warning, env_part = split_obs(step.observation)
            if flags.repeat_hint and env_part and _EMPTY_OBS_RE.fullmatch(env_part.strip()):
                hint_parts.append(_HINT_EMPTY_OBS)
            if hint_parts:
                block = "HINTS:\n" + "".join(hint_parts)
                warning = f"{warning}{block}" if warning else block
            if warning:
                stats.n_warnings += warning.count("WARNINGS:")
                if "Failed to parse tool call" in warning:
                    n_consec_parse_fail += 1
                else:
                    n_consec_parse_fail = 0
            else:
                n_consec_parse_fail = 0
            if env_part:
                stats.n_command_outputs += env_part.count("<command_output>")

            mask_warn, mask_env = world_flags(world_target)
            tr.add_env_feedback(markers, ((warning, mask_warn), (env_part, mask_env)))

            if (
                flags.max_parse_fail > 0
                and n_consec_parse_fail >= flags.max_parse_fail
                and not step.done
            ):
                step = await env_timed(env_time, env.step, _DONE_ACTION)
                final_reward = step.reward
                stats.close_turn(span_start, len(tr), executed=bool(cmd), done=bool(step.done))
                status = "truncated"
                abort_reason = "parse_fail"
                break

            stats.close_turn(span_start, len(tr), executed=bool(cmd), done=bool(step.done))

            if step.done:
                if getattr(step, "max_turns_forced", False):
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

    metadata = {
        "verifier_score": float(final_reward),
        "abort_reason": abort_reason,
        "num_turns": stats.num_turns,
        "max_turns": max_turns,
        "max_ctx": max_ctx,
        "prompt_len": len(prompt_ids),
        "response_len": len(tr),
        "turn_gen_lens": stats.turn_gen_lens,
        "turn_token_spans": stats.turn_token_spans,
        "turn_signals": stats.turn_signals,
        "n_warnings": stats.n_warnings,
        "n_command_outputs": stats.n_command_outputs,
        "n_commands": stats.n_commands,
        "n_command_tags": tr.text.count("<command>"),
        "n_parse_fail": tr.text.count("Failed to parse tool call"),
        "n_gen_trunc": stats.n_gen_trunc,
        "echo_world_loss_target": world_target,
        "echo_full_obs_count": tr.obs_token_count,
        "status": status,
        "env_time_s": round(env_time[0], 3),
        "episode_elapsed_s": round(time.monotonic() - t_episode0, 1),
        "env_never_started": _never_started(env),
        "cache_cached_tokens": stats.cache_cached_tokens,
        "cache_prompt_tokens": stats.cache_prompt_tokens,
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
