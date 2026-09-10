"""ECHO XML training protocol as a Harbor BaseAgent.

Runs the exact train-time interaction format (protocols/echo_xml) inside
`harbor run`: same system prompt, same <command>/<action>done</action> parse,
same observation text (WARNINGS + <command_output>, max_obs_chars truncation),
same turn semantics (user-role observations, per-turn token cap, episode
budget). Harbor still owns the environment and the verifier, so scores stay
on the harbor_terminus lane: score protocol label is
harbor_terminus/echo_xml_agent, never mixed with train-side ECHO XML
verifier numbers.

Dependency discipline: harbor/bin/harbor prepends the harbor_runtime tree,
which vendors tokenizers==0.23.1. This module must not import
transformers/sglang/tokenizers; the LLM call is a stdlib-urllib
chat.completions POST and token counts come from response `usage` only.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .xml import get_system_prompt, parse_action

try:
    from harbor.agents.base import BaseAgent
    from harbor.models.agent.context import AgentContext
except ImportError:
    # Unit tests run without harbor_runtime; harbor itself provides the real
    # classes when it imports this module by path.
    class BaseAgent:  # type: ignore[no-redef]
        def __init__(self, logs_dir, model_name=None, *args, **kwargs):
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name

    class AgentContext:  # type: ignore[no-redef]
        def __init__(self):
            self.n_input_tokens = None
            self.n_output_tokens = None
            self.metadata = None


def post_chat_completion(
    api_base: str, api_key: str, body: dict, timeout: int = 900
) -> dict:
    """Minimal OpenAI chat.completions POST (stdlib only)."""
    url = api_base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        raise RuntimeError(f"chat.completions HTTP {e.code} {url}: {raw[:500]}") from e


def format_exec_output(stdout: str | None, stderr: str | None) -> str:
    """stdout+stderr merge, identical to HarborTerminalEnv._exec_text."""
    out = stdout or ""
    if stderr:
        err = stderr.strip()
        if err:
            out = f"{out.rstrip()}\n{err}\n" if out else f"{err}\n"
    return out


def build_observation(warning: str, out: str | None, max_obs_chars: int) -> str:
    """Byte-identical to DockerTerminalEnv/HarborTerminalEnv.step obs layout.

    out=None means no command was executed (parse fail or done), which yields
    the empty <command_output> wrapper after the warning, as in training.
    """
    env_body = ""
    if out is not None:
        if len(out) > max_obs_chars:
            out = out[:max_obs_chars] + "\n...[truncated]...\n"
        env_body = out if out.endswith("\n") else out + "\n"
    env_output = f"<command_output>\n{env_body}</command_output>"
    return f"{warning}{env_output}" if warning else env_output


class EchoXmlAgent(BaseAgent):
    """Harbor agent form of the ECHO XML training protocol."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *args,
        api_base: str = "http://127.0.0.1:30000/v1",
        temperature: float = 0.6,
        max_turns: int = 16,
        max_new_tokens: int = 4096,
        max_obs_chars: int = 8000,
        user_role_obs: int = 1,
        enable_thinking: int = 0,
        episode_budget_s: float = 0,
        command_timeout: int = 120,
        request_timeout: int = 900,
        agent_version: str = "1.0.0",
        **kwargs,
    ):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self._api_base = api_base
        self._api_key = os.environ.get("OPENAI_API_KEY", "sk-harbor-local")
        self._temperature = float(temperature)
        self._max_turns = int(max_turns)
        self._max_new_tokens = int(max_new_tokens)
        self._max_obs_chars = int(max_obs_chars)
        self._user_role_obs = int(user_role_obs) == 1
        self._enable_thinking = int(enable_thinking) == 1
        self._episode_budget_s = float(episode_budget_s)
        self._command_timeout = int(command_timeout)
        self._request_timeout = int(request_timeout)
        self._agent_version = agent_version
        self._chat: list[dict[str, str]] = []
        self._total_input_tokens: int | None = None
        self._total_output_tokens: int | None = None

    @staticmethod
    def name() -> str:
        return "echo-xml"

    def version(self) -> str | None:
        return self._agent_version

    async def setup(self, environment) -> None:
        # No in-container setup; commands go through environment.exec().
        self._chat = []
        self._total_input_tokens = None
        self._total_output_tokens = None

    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        # Train-time reset(): system prompt and task ride in ONE user message.
        self._chat = [
            {
                "role": "user",
                "content": f"{get_system_prompt()}\n\nTask:\n{instruction.strip()}\n",
            }
        ]
        status, abort_reason = "truncated", "max_turns"
        n_turns = 0
        t0 = time.monotonic()
        try:
            for _ in range(self._max_turns):
                if self._episode_budget_s > 0 and (time.monotonic() - t0) > self._episode_budget_s:
                    status, abort_reason = "truncated", "time_budget"
                    break
                text, finish, usage = await self._generate()
                self._chat.append({"role": "assistant", "content": text})
                n_turns += 1
                self._track_usage(usage)
                if finish == "length":
                    # Train default (ECHO_GEN_LENGTH_CONTINUE=0): a turn hitting
                    # the per-turn cap kills the episode with abort=gen_length.
                    status, abort_reason = "truncated", "gen_length"
                    break
                cmd, done, warning = parse_action(text)
                out: str | None = None
                if cmd:
                    try:
                        r = await environment.exec(
                            command=cmd, timeout_sec=self._command_timeout
                        )
                        out = format_exec_output(r.stdout, r.stderr)
                    except Exception as exc:  # noqa: BLE001
                        out = f"[harbor_exec_error] {type(exc).__name__}: {exc}"
                obs = build_observation(warning, out, self._max_obs_chars)
                if self._user_role_obs:
                    # C1: env feedback is the next user message.
                    self._chat.append({"role": "user", "content": obs})
                else:
                    self._chat[-1]["content"] += obs
                if done:
                    status, abort_reason = "completed", "done"
                    break
        finally:
            if self._total_input_tokens is not None:
                context.n_input_tokens = self._total_input_tokens
            if self._total_output_tokens is not None:
                context.n_output_tokens = self._total_output_tokens
            context.metadata = {
                "n_turns": n_turns,
                "status": status,
                "abort_reason": abort_reason,
                "messages": self._chat,
            }

    def _served_model(self) -> str:
        # HARBOR_MODEL is litellm-style "provider/model" (e.g. openai/Qwen3-8B);
        # strip the provider like litellm does before hitting SGLang.
        name = self.model_name or ""
        return name.split("/", 1)[1] if "/" in name else name

    def _track_usage(self, usage: dict) -> None:
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if pt is not None:
            self._total_input_tokens = (self._total_input_tokens or 0) + int(pt)
        if ct is not None:
            self._total_output_tokens = (self._total_output_tokens or 0) + int(ct)

    async def _generate(self) -> tuple[str, str, dict]:
        body: dict[str, Any] = {
            "model": self._served_model(),
            "messages": self._chat,
            "temperature": self._temperature,
            "max_tokens": self._max_new_tokens,
            # Qwen3 thinking switch, same knob as train-side apply_chat_template.
            "chat_template_kwargs": {"enable_thinking": self._enable_thinking},
        }
        resp = await asyncio.to_thread(
            post_chat_completion,
            self._api_base,
            self._api_key,
            body,
            self._request_timeout,
        )
        choice = (resp.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        finish = str(choice.get("finish_reason") or "")
        return text, finish, resp.get("usage") or {}
