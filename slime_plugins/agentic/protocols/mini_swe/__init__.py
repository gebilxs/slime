"""Mini-SWE training harness (live generate / reward / env)."""

from __future__ import annotations

from .generate import (
    apply_episode_to_sample,
    build_env,
    generate,
    generate as generate_episode,
    run_mini_swe_episode,
)
from .mswea import parse_action
from .reward import reward_func

name = "mini_swe"
PROTOCOL_SCALESWE = "scaleswe"
PROTOCOL_SWEBENCH = "swebench"


class MiniSWEActionParser:
    parse = staticmethod(parse_action)
    __call__ = staticmethod(parse_action)


ActionParser = MiniSWEActionParser


def make_action_parser():
    return MiniSWEActionParser()


def make_episode(env, **kwargs):
    raise RuntimeError(
        "training episodes go through run_mini_swe_episode / generate; "
        "ProtocolEpisode is mock-only"
    )


async def reward(args, sample, **kwargs):
    return await reward_func(args, sample, **kwargs)


score = reward
compute_reward = reward_func


def get_metadata(sample):
    from core.tasks import instruction_for_task, resolve_task_path

    meta = dict(getattr(sample, "metadata", None) or {})
    try:
        path = resolve_task_path(sample)
    except Exception:
        path = None
    if path is not None:
        meta.setdefault("task_path", str(path))
        meta.setdefault("instruction", instruction_for_task(sample, path))
    return meta


def evaluability_check(metadata: dict) -> str | None:
    if not isinstance(metadata, dict):
        return "metadata_not_mapping"
    if not (metadata.get("task_id") or metadata.get("instance_id") or metadata.get("task_path")):
        return "missing_task_id"
    return None


__all__ = [
    "ActionParser",
    "MiniSWEActionParser",
    "PROTOCOL_SCALESWE",
    "PROTOCOL_SWEBENCH",
    "apply_episode_to_sample",
    "build_env",
    "compute_reward",
    "evaluability_check",
    "generate",
    "generate_episode",
    "get_metadata",
    "make_action_parser",
    "make_episode",
    "name",
    "parse_action",
    "reward",
    "reward_func",
    "run_mini_swe_episode",
    "score",
]
