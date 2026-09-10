"""ECHO generate: the slime entry point and the Sample adapter.

``generate`` is the slime hook (``--custom-generate-function-path
slime_plugins.agentic.protocols.echo_xml.generate.generate``); it lazy-imports slime so unit tests
and API-only eval never pull Megatron in. The episode itself is slime-free and
lives in ``slime_plugins.agentic.protocols.echo_xml.episode``.

``sample.reward`` is left unset. The verifier score goes in metadata so
``--custom-rm-path slime_plugins.agentic.protocols.echo_xml.reward.reward_func`` owns the scalar.

This module stays the public face of the protocol: the pieces below were split
out for readability, but every name callers already import is re-exported here.

  episode.py      turn loop, EpisodeFlags, Transcript, EpisodeStats
  chat_format.py  chat-template rendering and SGLang finish_reason plumbing
  env_factory.py  ENV_BACKEND -> mock | docker | harbor
  env_pool.py     shared thread pool and in-flight container semaphore
"""

from __future__ import annotations

import os

# Imported for its side effect on the module namespace: tests reach the clock
# through `slime_plugins.agentic.protocols.echo_xml.generate.time` to freeze episode wall time.
import time  # noqa: F401
from typing import Any, Awaitable, Callable

from core.environ import getenv
from core.chat_format import (
    apply_chat_template,
    chat_markers,
    ends_with_im_end,
    sglang_stop_matched,
)
from core.env_pool import _docker_sem, env_timed, env_to_thread
from core.episode import EpisodeResult
from core.episode import apply_episode_to_sample as _apply_to_sample
from core.transcript import Transcript
from .env_factory import build_env, instruction_for_task, resolve_task_path
from .episode import EpisodeFlags, EpisodeStats, run_echo_episode

TurnGenerate = Callable[[list[int], dict], Awaitable[dict]]

__all__ = [
    # slime-facing
    "generate",
    "apply_episode_to_sample",
    # episode core
    "run_echo_episode",
    "EpisodeResult",
    "EpisodeFlags",
    "EpisodeStats",
    "Transcript",
    "TurnGenerate",
    # env construction
    "build_env",
    "resolve_task_path",
    "instruction_for_task",
    # chat template / SGLang
    "apply_chat_template",
    "chat_markers",
    "ends_with_im_end",
    "sglang_stop_matched",
    # concurrency
    "env_timed",
    "env_to_thread",
]


def apply_episode_to_sample(sample, result: EpisodeResult, backend: str) -> Any:
    """Write episode fields onto a slime Sample. Does not set sample.reward."""
    return _apply_to_sample(
        sample,
        result,
        backend,
        backend_key="echo_env_backend",
        obs_count_attr="echo_full_obs_count",
    )


async def generate(args: Any, sample: Any, sampling_params: dict, evaluation: bool = False):
    """slime --custom-generate-function-path entry (imports slime lazily)."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post

    assert not isinstance(sample.prompt, list), "echo_xml expects a string prompt"
    state = GenerateState(args)
    tokenizer = state.tokenizer
    max_turns = int(getenv("MAX_TURNS", getattr(args, "echo_max_turns", 8) or 8))
    target = getattr(args, "echo_world_loss_target", None) or os.environ.get(
        "ECHO_WORLD_LOSS_TARGET", "env_only"
    )
    env, backend = build_env(sample, max_turns)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    async def generate_turn(input_ids: list[int], sp: dict) -> dict:
        return await post(
            url,
            {"input_ids": input_ids, "sampling_params": sp, "return_logprob": True},
        )

    sem = _docker_sem() if backend in ("docker", "harbor") else None

    async def _run():
        result = await run_echo_episode(
            prompt=sample.prompt,
            env=env,
            tokenizer=tokenizer,
            generate_turn=generate_turn,
            sampling_params=sampling_params,
            max_turns=max_turns,
            max_ctx=int(getenv("MAX_CONTEXT_LEN", "4096")),
            world_target=target,
            evaluation=evaluation,
            label=str(getattr(sample, "label", "") or ""),
        )
        return apply_episode_to_sample(sample, result, backend)

    if sem is None:
        return await _run()
    async with sem:
        return await _run()
