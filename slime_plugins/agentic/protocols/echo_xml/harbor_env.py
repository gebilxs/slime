"""ECHO XML env over Harbor DockerEnvironment (not Terminus, not harbor run).

Same observation layout as DockerTerminalEnv: WARNINGS + <command_output>.
Harbor only starts/execs/stops the task container. The train loop still parses
the first <command> and scores tests/test.sh at <action>done</action>.

harbor_runtime is appended to sys.path, never prepended. That tree vendors
tokenizers==0.23.1; SGLang/transformers 4.57 need the system 0.22.2.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .xml import StepResult, get_system_prompt, parse_action

logger = logging.getLogger(__name__)

HarborFactory = Callable[..., Any]

_PATH_LOCK = threading.Lock()
_OVERLAY_LOCK = threading.Lock()
_OVERLAY_PATH: Path | None = None
_LOOP: "_HarborLoop | None" = None
_LOOP_LOCK = threading.Lock()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "recipes" / "eval").is_dir() and (p / "core").is_dir():
            return p
    raise RuntimeError(f"cannot find manual_slime repo root from {here}")


def default_harbor_runtime() -> Path:
    raw = os.environ.get("HARBOR_RUNTIME", "").strip()
    if raw:
        return Path(raw)
    return _repo_root().parent / "0805_ICLR_Submit" / "pkgs" / "harbor_runtime"


def ensure_harbor_on_path() -> Path:
    """Append harbor_runtime. Never insert at sys.path[0]."""
    import sys

    root = default_harbor_runtime().resolve()
    s = str(root)
    with _PATH_LOCK:
        if s not in sys.path:
            sys.path.append(s)
    return root


def import_harbor() -> tuple[Any, Any, Any]:
    """DockerEnvironment, TaskConfig, TrialPaths from the Harbor tree."""
    root = ensure_harbor_on_path()
    if not (root / "harbor").is_dir():
        raise RuntimeError(f"HARBOR_RUNTIME has no harbor package: {root}")
    try:
        from harbor.environments.docker.docker import DockerEnvironment
        from harbor.models.task.config import TaskConfig
        from harbor.models.trial.paths import TrialPaths
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import Harbor from {root} (append-only sys.path): {exc}"
        ) from exc
    return DockerEnvironment, TaskConfig, TrialPaths


class _HarborLoop:
    """One event loop for all Harbor async calls in this process.

    DockerEnvironment uses class-level asyncio.Lock for image builds. Those
    locks are loop-bound, so reset/step/close must share a loop.
    """

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run, name="echo-harbor-loop", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=15):
            raise RuntimeError("Harbor asyncio loop failed to start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    def run(self, coro):
        loop = self._loop
        assert loop is not None
        return asyncio.run_coroutine_threadsafe(coro, loop).result()


def harbor_loop() -> _HarborLoop:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None:
            _LOOP = _HarborLoop()
        return _LOOP


def jump_compose_path() -> Path | None:
    """Compose overlay so task containers can reach the cluster jump proxy."""
    if os.environ.get("HARBOR_JUMP", "1") == "0":
        return None
    explicit = os.environ.get("HARBOR_JUMP_COMPOSE", "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"HARBOR_JUMP_COMPOSE not a file: {p}")
        return p
    url = os.environ.get("JUMP_PROXY_URL", "").strip()
    if not url:
        bundled = _repo_root() / "recipes" / "eval" / "compose_jump_proxy.yaml"
        return bundled if bundled.is_file() else None
    global _OVERLAY_PATH
    with _OVERLAY_LOCK:
        if _OVERLAY_PATH is not None and _OVERLAY_PATH.is_file():
            return _OVERLAY_PATH
        root = Path(os.environ.get("HARBOR_TRIAL_ROOT") or tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        path = root / "echo_harbor_jump_overlay.yaml"
        path.write_text(
            "services:\n"
            "  main:\n"
            "    network_mode: host\n"
            "    environment:\n"
            f'      http_proxy: "{url}"\n'
            f'      https_proxy: "{url}"\n'
            f'      HTTP_PROXY: "{url}"\n'
            f'      HTTPS_PROXY: "{url}"\n'
            f'      ALL_PROXY: "{url}"\n'
            "      no_proxy: 127.0.0.1,localhost,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn\n"
            "      NO_PROXY: 127.0.0.1,localhost,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn\n",
            encoding="utf-8",
        )
        _OVERLAY_PATH = path
        return path


def load_task_env_config(task_path: Path, image: str | None):
    _, TaskConfig, _ = import_harbor()
    toml_path = task_path / "task.toml"
    if toml_path.is_file():
        cfg = TaskConfig.model_validate_toml(
            toml_path.read_text(encoding="utf-8", errors="replace")
        )
    else:
        cfg = TaskConfig()
    env_cfg = cfg.environment
    if image:
        env_cfg.docker_image = image
    elif not env_cfg.docker_image:
        from .docker_env import resolve_docker_image

        hint = resolve_docker_image(task_path, fallback=False)
        if hint:
            env_cfg.docker_image = hint
    return env_cfg


@dataclass
class HarborTerminalEnv:
    instruction: str
    task_path: str | Path
    max_turns: int = 8
    max_obs_chars: int = 8000
    command_timeout: int = 120
    run_verifier: bool = True
    image: str | None = None
    docker_env_factory: HarborFactory | None = None

    def __post_init__(self) -> None:
        self.task_path = Path(self.task_path)
        self.turn = 0
        self._closed = False
        self._hb: Any | None = None
        self._trial_dir: Path | None = None
        self._started = False
        self._ever_started = False
        # Lazy start: compose up happens on first exec/verify, matching
        # DockerTerminalEnv. LAZY_CREATE=0 restores eager reset()-time start.
        self._lazy = os.environ.get("LAZY_CREATE", "1") == "1"

    @property
    def started(self) -> bool:
        return self._started

    @property
    def ever_started(self) -> bool:
        """Survives close(); used for episode stats after teardown."""
        return self._ever_started

    def _run(self, coro):
        return harbor_loop().run(coro)

    def reset(self) -> str:
        self.turn = 0
        self.close()
        self._closed = False
        trial_root = Path(
            os.environ.get("HARBOR_TRIAL_ROOT") or tempfile.gettempdir()
        )
        trial_root.mkdir(parents=True, exist_ok=True)
        self._trial_dir = Path(
            tempfile.mkdtemp(
                prefix=f"echo-hb-{self.task_path.name}-", dir=str(trial_root)
            )
        )
        env_dir = self.task_path / "environment"
        if not env_dir.is_dir():
            env_dir = self._trial_dir / "environment"
            env_dir.mkdir(parents=True, exist_ok=True)

        extra: list[Path] = []
        overlay = jump_compose_path()
        if overlay is not None:
            extra.append(overlay)

        session_id = f"echo-{self.task_path.name}-{uuid.uuid4().hex[:8]}"
        force_build = os.environ.get("HARBOR_FORCE_BUILD", "0") == "1"
        keep = os.environ.get("HARBOR_KEEP_CONTAINERS", "0") == "1"

        if self.docker_env_factory is not None:
            self._hb = self.docker_env_factory(
                environment_dir=env_dir,
                environment_name=self.task_path.name,
                session_id=session_id,
                trial_dir=self._trial_dir,
                extra_docker_compose=extra,
                image=self.image,
            )
        else:
            DockerEnvironment, _, TrialPaths = import_harbor()
            paths = TrialPaths(trial_dir=self._trial_dir)
            paths.mkdir()
            env_cfg = load_task_env_config(self.task_path, self.image)
            kwargs: dict[str, Any] = {
                "environment_dir": env_dir,
                "environment_name": self.task_path.name,
                "session_id": session_id,
                "trial_paths": paths,
                "task_env_config": env_cfg,
                "keep_containers": keep,
            }
            if extra:
                kwargs["extra_docker_compose"] = extra
            self._hb = DockerEnvironment(**kwargs)

        if not self._lazy:
            self._start(force_build=force_build)
        return f"{get_system_prompt()}\n\nTask:\n{self.instruction.strip()}\n"

    def _start(self, force_build: bool) -> None:
        assert self._hb is not None
        self._run(self._hb.start(force_build=force_build))
        self._started = True
        self._ever_started = True
        logger.info(
            "harbor sandbox ready task=%s trial=%s image=%s",
            self.task_path.name,
            self._trial_dir,
            self.image,
        )

    def _ensure_started(self) -> None:
        if self._hb is None:
            raise RuntimeError("harbor env not reset()")
        if self._started:
            return
        force_build = os.environ.get("HARBOR_FORCE_BUILD", "0") == "1"
        self._start(force_build=force_build)

    def _exec_text(self, cmd: str, *, timeout: int | None = None) -> str:
        self._ensure_started()
        timeout = timeout or self.command_timeout
        result = self._run(self._hb.exec(cmd, timeout_sec=timeout))
        out = result.stdout or ""
        if result.stderr:
            err = result.stderr.strip()
            if err:
                out = f"{out.rstrip()}\n{err}\n" if out else f"{err}\n"
        return out

    def step(self, action: str) -> StepResult:
        if self._hb is None:
            raise RuntimeError("harbor env not reset()")
        self.turn += 1
        cmd, done, warning = parse_action(action)
        env_body = ""
        reward = 0.0

        if cmd:
            try:
                out = self._exec_text(cmd, timeout=self.command_timeout)
            except Exception as exc:  # noqa: BLE001
                out = f"[harbor_exec_error] {type(exc).__name__}: {exc}"
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

    def _verify(self) -> float:
        self._ensure_started()
        tests = self.task_path / "tests"
        try:
            if hasattr(self._hb, "upload_dir"):
                self._run(self._hb.upload_dir(tests, "/tests"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("harbor upload tests failed: %s", exc)
            return 0.0
        vt = int(os.environ.get("VERIFY_TIMEOUT", "600"))
        inner_vt = max(30, vt - 30)
        script = (
            "mkdir -p /logs/verifier && "
            "if command -v timeout >/dev/null 2>&1; then "
            f"timeout -k 10 {inner_vt} bash /tests/test.sh "
            "> /logs/verifier/test.stdout 2> /logs/verifier/test.stderr; "
            "else bash /tests/test.sh "
            "> /logs/verifier/test.stdout 2> /logs/verifier/test.stderr; fi; "
            "echo $? > /logs/verifier/exit_code; "
            "if [ -f /logs/verifier/reward.txt ]; then "
            "tail -n 1 /logs/verifier/reward.txt; "
            "else ec=$(cat /logs/verifier/exit_code); "
            '[ "$ec" = 0 ] && echo 1 || echo 0; fi'
        )
        try:
            out = self._exec_text(script, timeout=vt)
            line = (out or "0").strip().splitlines()[-1].strip()
            return float(line)
        except Exception as exc:  # noqa: BLE001
            logger.warning("harbor verifier failed: %s", exc)
            return 0.0

    def close(self) -> None:
        hb, trial = self._hb, self._trial_dir
        was_started = self._started
        self._hb = None
        self._trial_dir = None
        self._started = False
        self._closed = True
        if hb is not None and was_started:
            try:
                delete = os.environ.get("HARBOR_KEEP_CONTAINERS", "0") != "1"
                self._run(hb.stop(delete=delete))
            except Exception as exc:  # noqa: BLE001
                logger.warning("harbor stop failed: %s", exc)
        if trial is not None and os.environ.get("HARBOR_KEEP_TRIALS", "0") != "1":
            shutil.rmtree(trial, ignore_errors=True)
