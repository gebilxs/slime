"""mini-SWE docker env: mini-swe-agent semantics over the ECHO HTTP sidecar.

Execution model mirrors minisweagent.environments.docker: every action runs
in a fresh subshell (the sidecar /exec wraps bash -lc, merges stderr into
stdout, enforces a timeout kill), cwd defaults to /workspace. Container
creation is lazy — episodes that never emit a valid action pay zero sandbox
cost (RL_INFRA.md §3). Verifier mirrors echo_xml.docker_env._verify
(/oracle/tests/test.sh -> reward.txt or exit code; binary 0/1).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.environ import getenv

from core.sandbox import http_json, load_sandbox_urls
from .mswea import (
    MiniStepResult,
    check_submitted,
    format_error_text,
    parse_action,
    render_observation,
)

logger = logging.getLogger(__name__)

HttpFn = Callable[[str, str, dict | None, int], dict]

_RR_LOCK = threading.Lock()
_RR_IDX = 0


@dataclass
class DockerSweEnv:
    instruction: str
    task_path: str | Path
    max_turns: int = 16
    max_obs_chars: int = 10000
    command_timeout: int = 120
    run_verifier: bool = True
    image: str | None = None
    urls: list[str] | None = None
    http: HttpFn | None = None
    dataset: str | None = None

    def __post_init__(self) -> None:
        self.task_path = Path(self.task_path)
        self.turn = 0
        self._closed = False
        self._http = self.http or http_json
        self._urls = list(self.urls) if self.urls is not None else load_sandbox_urls()
        self._base: str | None = None
        self._cid: str | None = None
        self._ever_started = False
        self._lazy = os.environ.get("MINI_SWE_LAZY_CREATE", "1") == "1"

    @property
    def started(self) -> bool:
        return self._cid is not None

    @property
    def ever_started(self) -> bool:
        """Survives close(); used for episode stats after teardown."""
        return self._ever_started

    def reset(self) -> str:
        """No container yet (lazy); returns nothing — prompt build is generate's job."""
        self.turn = 0
        self.close()
        self._closed = False
        if not self._lazy:
            self._ensure_started()
        return self.instruction

    def _ensure_started(self) -> None:
        if self._cid:
            return
        errors: list[str] = []
        global _RR_IDX
        create_timeout = int(getenv("DOCKER_CREATE_TIMEOUT", "960"))
        wait_s = int(getenv("DOCKER_CREATE_WAIT", "20"))
        deadline = time.time() + create_timeout
        while True:
            order = list(range(len(self._urls)))
            with _RR_LOCK:
                start = _RR_IDX % len(self._urls)
                _RR_IDX += 1
            order = order[start:] + order[:start]
            n_full = 0
            for off in order:
                base = self._urls[off]
                body = {
                    "task_path": str(self.task_path),
                    "name": f"echo-{self.task_path.name}-{uuid.uuid4().hex[:10]}",
                    "wait_s": wait_s,
                }
                if self.image:
                    body["image"] = self.image
                try:
                    c = self._http("POST", f"{base}/create", body, create_timeout)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{base}: {type(exc).__name__}: {exc}")
                    continue
                if c.get("ok"):
                    self._base = base
                    self._cid = c["id"]
                    self._ever_started = True
                    logger.info(
                        "mini_swe sandbox ready host=%s id=%s task=%s image=%s",
                        base,
                        self._cid[:12],
                        self.task_path.name,
                        c.get("image"),
                    )
                    return
                err = str(c.get("error") or c)
                errors.append(f"{base}: create {err[:200]}")
                if "no free sandbox" in err:
                    n_full += 1
            if time.time() >= deadline:
                break
            time.sleep(2.0 if n_full == len(self._urls) else 1.0)
            errors = errors[-12:]
        raise RuntimeError("docker sandbox create failed: " + " | ".join(errors[:8]))

    def _exec_full(self, cmd: str, *, timeout: int | None = None) -> dict:
        self._ensure_started()
        assert self._base and self._cid
        return self._http(
            "POST",
            f"{self._base}/exec",
            {"id": self._cid, "cmd": cmd, "timeout": timeout or self.command_timeout},
            (timeout or self.command_timeout) + 30,
        )

    def _exec(self, cmd: str, *, timeout: int | None = None) -> str:
        r = self._exec_full(cmd, timeout=timeout)
        return r.get("stdout") or ""

    def step(self, action_text: str, gen_truncated: bool = False) -> MiniStepResult:
        """One agent turn: parse -> execute -> observation / submit / verify.

        gen_truncated=True marks the model turn as finish_reason=length: the
        format-error feedback then uses upstream's dedicated length message
        (and a complete action, if one made it in, is still executed — that
        is what upstream's regex parse does on truncated content).
        """
        self.turn += 1
        actions, _span = parse_action(action_text)

        if len(actions) != 1:
            obs = format_error_text(len(actions), truncated=gen_truncated)
            done = False
            reward = 0.0
            max_turns_forced = False
            if self.turn >= self.max_turns:
                done = True
                max_turns_forced = True
                reward = self._verify_or_zero()
            return MiniStepResult(
                observation=obs,
                done=done,
                reward=reward,
                command=None,
                n_actions=len(actions),
                is_format_error=True,
                max_turns_forced=max_turns_forced,
            )

        cmd = actions[0]
        try:
            r = self._exec_full(cmd, timeout=self.command_timeout)
            out = r.get("stdout") or ""
            rc = int(r.get("exit_code", 0) or 0)
            exc_info = ""
        except Exception as exc:  # noqa: BLE001
            out, rc = f"[docker_exec_error] {type(exc).__name__}: {exc}", -1
            exc_info = f"An error occurred while executing the command: {exc}"

        submitted, submission = check_submitted(out, rc)
        done = submitted or self.turn >= self.max_turns
        max_turns_forced = (not submitted) and self.turn >= self.max_turns
        reward = 0.0
        if done:
            reward = self._verify_or_zero()

        obs = render_observation(out, rc, exc_info, max_chars=self.max_obs_chars)
        return MiniStepResult(
            observation=obs,
            done=done,
            reward=reward,
            command=cmd,
            n_actions=1,
            is_format_error=False,
            submitted=submitted,
            max_turns_forced=max_turns_forced,
        )

    def force_done(self) -> float:
        """Verify-only exit (format-error cap / time budget): keep partial credit."""
        return self._verify_or_zero()

    def _verify_or_zero(self) -> float:
        if not self.run_verifier:
            return 0.0
        if not (self.task_path / "tests" / "test.sh").is_file():
            return 0.0
        return self._verify()

    def _verify(self) -> float:
        vt = int(getenv("VERIFY_TIMEOUT", "600"))
        inner_vt = max(30, vt - 30)
        script = (
            "mkdir -p /logs/verifier && "
            "if command -v timeout >/dev/null 2>&1; then "
            f"timeout -k 10 {inner_vt} bash /oracle/tests/test.sh "
            "> /logs/verifier/test.stdout 2> /logs/verifier/test.stderr; "
            "else bash /oracle/tests/test.sh "
            "> /logs/verifier/test.stdout 2> /logs/verifier/test.stderr; fi; "
            "echo $? > /logs/verifier/exit_code; "
            "if [ -f /logs/verifier/reward.txt ]; then "
            "tail -n 1 /logs/verifier/reward.txt; "
            "else ec=$(cat /logs/verifier/exit_code); "
            "[ \"$ec\" = 0 ] && echo 1 || echo 0; fi"
        )
        try:
            out = self._exec(script, timeout=vt)
            line = (out or "0").strip().splitlines()[-1].strip()
            return float(line)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mini_swe verifier failed: %s", exc)
            return 0.0

    def close(self) -> None:
        base, cid = self._base, self._cid
        self._base = None
        self._cid = None
        self._closed = True
        if not base or not cid:
            return
        try:
            self._http("POST", f"{base}/destroy", {"id": cid}, 60)
        except Exception as exc:  # noqa: BLE001
            logger.warning("docker destroy failed id=%s: %s", cid[:12], exc)
