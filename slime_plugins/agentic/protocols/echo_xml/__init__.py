"""Echo XML training harness (live generate / reward / env)."""

from __future__ import annotations

from .env_factory import build_env
from .generate import apply_episode_to_sample, generate
from .reward import reward_func
from .xml import parse_action

name = "echo_xml"
PROTOCOL = name


class EchoXMLActionParser:
    parse = staticmethod(parse_action)
    __call__ = staticmethod(parse_action)


ActionParser = EchoXMLActionParser


def make_action_parser():
    return EchoXMLActionParser()


def make_episode(env, **kwargs):
    raise RuntimeError(
        "training episodes go through generate; ProtocolEpisode is mock-only"
    )


async def reward(args, sample, **kwargs):
    return await reward_func(args, sample, **kwargs)


score = reward
compute_reward = reward_func
generate_episode = generate


def evaluability_check(metadata: dict) -> str | None:
    if not isinstance(metadata, dict):
        return "metadata_not_mapping"
    if not (metadata.get("task_id") or metadata.get("instance_id")):
        return "missing_task_id"
    return None


__all__ = [
    "ActionParser",
    "EchoXMLActionParser",
    "PROTOCOL",
    "apply_episode_to_sample",
    "build_env",
    "compute_reward",
    "evaluability_check",
    "generate",
    "generate_episode",
    "make_action_parser",
    "make_episode",
    "name",
    "parse_action",
    "reward",
    "reward_func",
    "score",
]
