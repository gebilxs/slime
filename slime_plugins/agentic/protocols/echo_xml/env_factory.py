"""Sandbox selection for ECHO episodes: ENV_BACKEND -> env instance.

Backends are imported lazily so a mock-backend run never pulls in the docker
or harbor client stacks.
"""

from __future__ import annotations

import os

from core.environ import getenv
# Task location is harness-neutral; re-exported here for existing callers.
from core.tasks import instruction_for_task, resolve_task_path

__all__ = ["build_env", "resolve_task_path", "instruction_for_task"]


def build_env(sample, max_turns: int):
    backend = getenv("ENV_BACKEND", "mock")
    if backend == "mock":
        from .mock_env import MockTerminalEnv

        return MockTerminalEnv(sample.prompt, max_turns=max_turns), backend
    if backend == "docker":
        from .docker_env import DockerTerminalEnv, resolve_docker_image

        task_path = resolve_task_path(sample)
        meta = dict(getattr(sample, "metadata", None) or {})
        image = meta.get("image") or getenv("DOCKER_IMAGE")
        if not image:
            image = resolve_docker_image(task_path)
        env = DockerTerminalEnv(
            instruction=instruction_for_task(sample, task_path),
            task_path=task_path,
            max_turns=max_turns,
            max_obs_chars=int(getenv("MAX_OBS_CHARS", "8000")),
            command_timeout=int(getenv("COMMAND_TIMEOUT", "120")),
            run_verifier=getenv("RUN_VERIFIER", "1") != "0",
            image=image or None,
            dataset=meta.get("dataset"),
        )
        return env, backend
    if backend == "harbor":
        from .harbor_env import HarborTerminalEnv

        task_path = resolve_task_path(sample)
        meta = dict(getattr(sample, "metadata", None) or {})
        # Explicit only. Manifest/task.toml docker_image is resolved inside HarborTerminalEnv
        # so a Harbor-expected tag in task.toml is not overwritten by a registry source tag.
        image = meta.get("image") or getenv("DOCKER_IMAGE") or None
        env = HarborTerminalEnv(
            instruction=instruction_for_task(sample, task_path),
            task_path=task_path,
            max_turns=max_turns,
            max_obs_chars=int(getenv("MAX_OBS_CHARS", "8000")),
            command_timeout=int(getenv("COMMAND_TIMEOUT", "120")),
            run_verifier=getenv("RUN_VERIFIER", "1") != "0",
            image=image or None,
        )
        return env, backend
    raise NotImplementedError(
        f"ENV_BACKEND={backend!r} unsupported in manual_slime v1; use mock|docker|harbor"
    )
