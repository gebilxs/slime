"""Deterministic terminal stub for ECHO mask / generate tests (no Docker)."""

from __future__ import annotations

import hashlib

from .xml import StepResult, parse_action


class MockTerminalEnv:
    """Same observation layout as docker env: WARNINGS + <command_output>."""

    def __init__(self, task_prompt: str, max_turns: int = 4, *, verifier_on_done: float = 1.0):
        self.task_prompt = task_prompt
        self.max_turns = max_turns
        self.verifier_on_done = verifier_on_done
        self.turn = 0
        self._done = False
        self.commands: list[str] = []

    def reset(self) -> str:
        self.turn = 0
        self._done = False
        self.commands = []
        return (
            "You are a terminal agent. Solve the task with "
            "<command>…</command> or finish with <action>done</action>.\n"
            f"Task: {self.task_prompt}"
        )

    def step(self, action: str) -> StepResult:
        self.turn += 1
        cmd, parse_done, warning = parse_action(action)
        if cmd:
            self.commands.append(cmd)

        digest = hashlib.md5((cmd or action).encode("utf-8", errors="ignore")).hexdigest()[:8]
        env_body = ""
        reward = 0.0
        max_turns_forced = False

        if cmd:
            env_body = f"stdout: listed /tmp/{digest}\nstderr:\nexit_code=0\n"

        done = parse_done or self.turn >= self.max_turns
        if done:
            max_turns_forced = (not parse_done) and self.turn >= self.max_turns
            if parse_done:
                env_body = env_body or f"exit_code=0\nverifier=pass\ndigest={digest}\n"
                reward = float(self.verifier_on_done)

        env_output = f"<command_output>\n{env_body}</command_output>"
        observation = f"{warning}{env_output}" if warning else env_output
        self._done = done
        return StepResult(
            observation=observation,
            warning=warning,
            env_output=env_output,
            done=done,
            reward=reward,
            max_turns_forced=max_turns_forced,
        )

    def close(self) -> None:
        return None
