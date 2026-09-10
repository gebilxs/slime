"""ECHO docker env: Gym reset/step over the local dockerd HTTP sidecar.

No E2B. No Brainbox. URLs come from DOCKER_SANDBOX_URLS (or a single
DOCKER_SANDBOX_URL). Image tags come from metadata, DOCKER_IMAGE,
or a jsonl manifest — never a hardcoded 0805 path.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Image resolution and sidecar HTTP are harness-neutral; re-exported here so
# existing importers of this module keep working.
from core.environ import getenv
from core.sandbox import (
    HttpFn,
    http_json,
    load_image_manifest,
    load_sandbox_urls,
    reset_manifest_cache,
    resolve_docker_image,
)
from .xml import StepResult, get_system_prompt, paper_format, parse_action

logger = logging.getLogger(__name__)

__all__ = [
    "DockerTerminalEnv",
    "reset_docker_client_state",
    "partial_reward_enabled",
    "partial_bonus",
    "partial_datasets",
    "fraction_from_ctrf",
    "fraction_from_pytest_stdout",
    # re-exported from core.sandbox for existing importers
    "HttpFn",
    "http_json",
    "load_image_manifest",
    "load_sandbox_urls",
    "resolve_docker_image",
]

_RR_LOCK = threading.Lock()
_RR_IDX = 0


def reset_docker_client_state() -> None:
    global _RR_IDX
    with _RR_LOCK:
        _RR_IDX = 0
    reset_manifest_cache()


def partial_reward_enabled() -> bool:
    """Partial-progress reward switch (yaml protocol.partial_reward)."""
    return os.environ.get("ECHO_PARTIAL_REWARD", "0").strip().lower() in ("1", "true", "yes", "on")


def partial_bonus() -> float:
    """Bonus added when ALL tests pass (SETA paper: 0.2)."""
    return float(os.environ.get("ECHO_PARTIAL_BONUS", "0.2"))


def partial_datasets() -> set[str] | None:
    """Datasets partial reward applies to (yaml protocol.partial_datasets).

    None (unset/empty) = apply to every dataset (back-compat). Otherwise only
    the listed datasets get partial credit; others keep their binary reward.
    """
    raw = os.environ.get("ECHO_PARTIAL_DATASETS", "").strip()
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def fraction_from_ctrf(raw: str, weights: dict | None = None) -> float | None:
    """Parse a pytest CTRF report into a (optionally weighted) pass fraction.

    Skipped/pending tests are excluded from the denominator. weights keys are
    bare test names (SETA weights.json); ctrf names may carry a path prefix
    ("tests/test_outputs.py::test_foo") — matched on the "::"-suffix.
    """
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    tests = (data.get("results") or {}).get("tests") or []
    if not tests:
        return None
    weights = weights or {}
    total = 0.0
    got = 0.0
    for t in tests:
        status = str(t.get("status") or "").lower()
        if status in ("skipped", "skip", "pending"):
            continue
        name = str(t.get("name") or "")
        short = name.split("::")[-1].strip()
        w = float(weights.get(short, 1.0)) if weights else 1.0
        total += w
        if status == "passed":
            got += w
    if total <= 0:
        return None
    return got / total


def fraction_from_pytest_stdout(out: str) -> float | None:
    """Parse `pytest -v` per-test lines into an unweighted pass fraction."""
    hits = re.findall(r"::\S+?\s+(PASSED|FAILED|ERROR|XFAIL|XPASS)", out)
    if not hits:
        return None
    n_eval = sum(1 for h in hits if h in ("PASSED", "FAILED", "ERROR", "XPASS"))
    if n_eval == 0:
        return None
    n_pass = sum(1 for h in hits if h == "PASSED")
    return n_pass / n_eval


@dataclass
class DockerTerminalEnv:
    instruction: str
    task_path: str | Path
    max_turns: int = 8
    max_obs_chars: int = 8000
    command_timeout: int = 120
    run_verifier: bool = True
    image: str | None = None
    urls: list[str] | None = None
    http: HttpFn | None = None
    # sample.metadata["dataset"]; used to scope partial reward per dataset.
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
        # Lazy create: the container is started on first _exec/_verify, so
        # episodes that never emit a valid command pay zero sandbox cost.
        # LAZY_CREATE=0 restores eager reset()-time creation.
        self._lazy = getenv("LAZY_CREATE", "1") == "1"

    @property
    def started(self) -> bool:
        return self._cid is not None

    @property
    def ever_started(self) -> bool:
        """Survives close(); used for episode stats after teardown."""
        return self._ever_started

    def reset(self) -> str:
        self.turn = 0
        self.close()
        self._closed = False
        if not self._lazy:
            self._ensure_started()
        return f"{get_system_prompt()}\n\nTask:\n{self.instruction.strip()}\n"

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
                        "docker sandbox ready host=%s id=%s task=%s image=%s",
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
            {
                "id": self._cid,
                "cmd": cmd,
                "timeout": timeout or self.command_timeout,
            },
            (timeout or self.command_timeout) + 30,
        )

    def _exec(self, cmd: str, *, timeout: int | None = None) -> str:
        r = self._exec_full(cmd, timeout=timeout)
        return r.get("stdout") or ""

    def step(self, action: str) -> StepResult:
        self.turn += 1
        cmd, done, warning = parse_action(action)
        env_body = ""
        reward = 0.0

        if cmd:
            if paper_format():
                # Official-harness obs: "Command '<cmd>' <status>. Output: <out>\n\n(exit_code=N)"
                try:
                    r = self._exec_full(cmd, timeout=self.command_timeout)
                    out = r.get("stdout") or ""
                    err = r.get("stderr") or ""
                    rc = int(r.get("exit_code", 0) or 0)
                except Exception as exc:  # noqa: BLE001
                    out, err, rc = f"[docker_exec_error] {type(exc).__name__}: {exc}", "", -1
                if err:
                    out = out + "\n" + err
                if len(out) > self.max_obs_chars:
                    out = out[: self.max_obs_chars] + "\n...[truncated]...\n"
                status = "executed successfully" if rc == 0 else "failed"
                env_body = f"Command '{cmd}' {status}. Output: {out}\n\n(exit_code={rc})"
            else:
                try:
                    out = self._exec(cmd, timeout=self.command_timeout)
                except Exception as exc:  # noqa: BLE001
                    out = f"[docker_exec_error] {type(exc).__name__}: {exc}"
                if len(out) > self.max_obs_chars:
                    out = out[: self.max_obs_chars] + "\n...[truncated]...\n"
                env_body = out if out.endswith("\n") else out + "\n"

        parse_done = done
        max_turns_forced = False
        if parse_done or self.turn >= self.max_turns:
            max_turns_forced = (not parse_done) and self.turn >= self.max_turns
            done = True
            if self.run_verifier and (self.task_path / "tests" / "test.sh").is_file():
                reward = self._verify()

        if paper_format():
            env_output = env_body  # official-harness format, no <command_output> wrapper
        else:
            env_output = f"<command_output>\n{env_body}</command_output>"
        observation = f"{warning}{env_output}" if warning else env_output
        return StepResult(
            observation=observation,
            warning=warning,
            env_output=env_output,
            done=done,
            reward=float(reward),
            max_turns_forced=max_turns_forced,
        )

    def _partial_fraction(self) -> float | None:
        """Partial-progress fraction in [0,1], or None when no per-test data
        is available (callers fall back to the binary reward.txt)."""
        try:
            raw = (self._exec("cat /logs/verifier/ctrf.json 2>/dev/null || true", timeout=60) or "").strip()
        except Exception:  # noqa: BLE001
            raw = ""
        if raw:
            weights = None
            wj = self.task_path / "weights.json" if self.task_path else None
            if wj is not None and wj.is_file():
                try:
                    weights = json.loads(wj.read_text())
                except Exception:  # noqa: BLE001
                    weights = None
            frac = fraction_from_ctrf(raw, weights)
            if frac is not None:
                return frac
        try:
            out = self._exec("cat /logs/verifier/test.stdout 2>/dev/null || true", timeout=60) or ""
        except Exception:  # noqa: BLE001
            return None
        return fraction_from_pytest_stdout(out)

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
            binary = float(line)
        except Exception as exc:  # noqa: BLE001
            logger.warning("docker verifier failed: %s", exc)
            return 0.0
        if not partial_reward_enabled():
            return binary
        allow = partial_datasets()
        if allow is not None and (self.dataset or "") not in allow:
            return binary
        frac = self._partial_fraction()
        if frac is None:
            return binary
        return frac + (partial_bonus() if frac >= 0.999 else 0.0)

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
