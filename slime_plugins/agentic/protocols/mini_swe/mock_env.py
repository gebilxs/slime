"""Deterministic mini-SWE env stub for tests and mock-backend smoke (no Docker)."""

from __future__ import annotations

import hashlib

from .mswea import (
    MiniStepResult,
    check_submitted,
    format_error_text,
    parse_action,
    render_observation,
)


class MockSweEnv:
    """Same observation layout as DockerSweEnv; no containers."""

    def __init__(self, task_prompt: str, max_turns: int = 4, *, verifier_on_done: float = 1.0):
        self.task_prompt = task_prompt
        self.max_turns = max_turns
        self.verifier_on_done = verifier_on_done
        self.turn = 0
        self.commands: list[str] = []
        self._ever_started = True

    @property
    def ever_started(self) -> bool:
        return self._ever_started

    def reset(self) -> str:
        self.turn = 0
        self.commands = []
        return self.task_prompt

    def step(self, action_text: str, gen_truncated: bool = False) -> MiniStepResult:
        self.turn += 1
        actions, _span = parse_action(action_text)

        if len(actions) != 1:
            done = self.turn >= self.max_turns
            return MiniStepResult(
                observation=format_error_text(len(actions), truncated=gen_truncated),
                done=done,
                reward=0.0,
                n_actions=len(actions),
                is_format_error=True,
                max_turns_forced=done,
            )

        cmd = actions[0]
        self.commands.append(cmd)
        digest = hashlib.md5(cmd.encode("utf-8", errors="ignore")).hexdigest()[:8]
        out = f"mock: ran `{cmd}`\nstdout: listed /tmp/{digest}\n"
        rc = 0
        submitted, _submission = check_submitted(out, rc)
        if cmd.strip() in ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",):
            out = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nmock submission\n"
            submitted = True

        done = submitted or self.turn >= self.max_turns
        reward = float(self.verifier_on_done) if submitted else 0.0
        return MiniStepResult(
            observation=render_observation(out, rc, max_chars=10000),
            done=done,
            reward=reward,
            command=cmd,
            n_actions=1,
            submitted=submitted,
            max_turns_forced=(not submitted) and done,
        )

    def force_done(self) -> float:
        return 0.0

    def close(self) -> None:
        return None
