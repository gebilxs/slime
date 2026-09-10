"""Harbor JobConfig YAML.

Harbor talks to an OpenAI HTTP endpoint itself. This module only writes the
job file; it does not start SGLang and does not subclass slime.agent.harness.
Which agent goes in the file is decided by evalkit.harness, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit import harness as harness_registry
from evalkit.harness import HarnessSpec, normalize_import_path

__all__ = [
    "HarborJobSpec",
    "build_config",
    "emit_yaml",
    "dump_yaml",
    "list_task_dirs",
    "normalize_import_path",
    "resolve_harness",
    "write_job_config",
]


def emit_yaml(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{sp}{key}:")
                lines.append(emit_yaml(value, indent + 1))
            elif isinstance(value, bool):
                lines.append(f"{sp}{key}: {'true' if value else 'false'}")
            elif value is None:
                lines.append(f"{sp}{key}: null")
            elif isinstance(value, (int, float)):
                lines.append(f"{sp}{key}: {value}")
            else:
                text = str(value).replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{sp}{key}: "{text}"')
        return "\n".join(lines) + ("\n" if indent == 0 else "")
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, dict):
                lines.append(f"{sp}-")
                lines.append(emit_yaml(item, indent + 1))
            else:
                lines.append(f"{sp}- {emit_yaml(item, 0).strip()}")
        return "\n".join(lines)
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if obj is None:
        return "null"
    if isinstance(obj, (int, float)):
        return str(obj)
    text = str(obj).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def dump_yaml(obj: dict) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(obj, sort_keys=False)
    except ImportError:
        return emit_yaml(obj)


@dataclass
class HarborJobSpec:
    job_name: str
    jobs_dir: str
    tasks_root: str
    n_attempts: int = 1
    n_concurrent: int = 4
    # Preferred: a registry key from evalkit.harness (echo_xml, terminus-2,
    # oracle). `agent` / `agent_import_path` remain for existing call sites.
    harness: str = ""
    agent: str = "terminus-2"
    agent_import_path: str = ""  # custom BaseAgent path; wins over agent
    model: str = "openai/Qwen3-8B"
    api_base: str = "http://127.0.0.1:30000/v1"
    temperature: float = 0.6
    # None means "whatever the harness declares" (echo_xml 16, terminus-2 100),
    # so the registry stays the single source of turn defaults.
    max_turns: int | None = None
    # EchoXmlAgent kwargs (emitted only when agent_import_path is set).
    max_new_tokens: int = 4096
    max_obs_chars: int = 8000
    user_role_obs: int = 1
    enable_thinking: int = 0
    episode_budget_s: float = 0
    parser_name: str = "xml"
    enable_summarize: bool = True
    record_terminal_session: bool = False
    # Terminus-2 sizes its own requests from model_info, so max_input +
    # max_output must fit the server's context window. The old 32768+8192
    # exceeded SGLang's --context-length 32768, and every call came back
    # finish_reason=length ("hit max_tokens limit. Response was truncated"),
    # which killed summarization and drove TB2 to ~0.
    context_length: int = 32768
    max_input_tokens: int = 24576
    max_output_tokens: int = 8192
    env_type: str = "docker"
    force_build: bool = False
    dataset_fallback: bool = False
    dataset: str = "terminal-bench@2.0"
    debug: bool = False
    extra_docker_compose: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)


def list_task_dirs(tasks_root: str | Path, task_filter: list[str] | None = None) -> list[Path]:
    root = Path(tasks_root)
    if task_filter:
        return [root / name for name in task_filter]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "task.toml").is_file())


def resolve_harness(spec: HarborJobSpec) -> HarnessSpec:
    """Pick the harness: explicit `harness` key, then import path, then `agent`."""
    if spec.harness:
        return harness_registry.get(spec.harness)
    if spec.agent_import_path:
        return harness_registry.get(spec.agent_import_path)
    return harness_registry.get(spec.agent)


def assert_token_budget(spec: HarborJobSpec) -> None:
    """Terminus-2 must be able to ask for max_output on a full-ish context."""
    total = spec.max_input_tokens + spec.max_output_tokens
    if total > spec.context_length:
        raise ValueError(
            f"token budget {spec.max_input_tokens} in + {spec.max_output_tokens} out "
            f"= {total} exceeds context_length {spec.context_length}. Terminus-2 would "
            "get finish_reason=length on every call. Lower max_input_tokens or raise "
            "CONTEXT_LEN in recipes/eval/serve_sglang.sh to match."
        )


def build_config(spec: HarborJobSpec) -> dict[str, Any]:
    harness = resolve_harness(spec)
    if harness.kind == harness_registry.KIND_BUILTIN:
        assert_token_budget(spec)
    agents = [
        harness_registry.agent_entry(
            harness,
            model=spec.model,
            api_base=spec.api_base,
            temperature=spec.temperature,
            max_turns=spec.max_turns,
            echo_kwargs={
                "max_new_tokens": spec.max_new_tokens,
                "max_obs_chars": spec.max_obs_chars,
                "user_role_obs": spec.user_role_obs,
                "enable_thinking": spec.enable_thinking,
                "episode_budget_s": spec.episode_budget_s,
            },
            terminus_kwargs={
                "parser_name": spec.parser_name,
                "enable_summarize": spec.enable_summarize,
                "record_terminal_session": spec.record_terminal_session,
                "model_info": {
                    "max_input_tokens": spec.max_input_tokens,
                    "max_output_tokens": spec.max_output_tokens,
                },
            },
        )
    ]

    cfg: dict[str, Any] = {
        "job_name": spec.job_name,
        "jobs_dir": str(spec.jobs_dir),
        "n_attempts": spec.n_attempts,
        "n_concurrent_trials": spec.n_concurrent,
        "debug": bool(spec.debug),
        "environment": {
            "type": spec.env_type,
            "force_build": bool(spec.force_build),
            "delete": True,
        },
        "agents": agents,
    }
    extra = [p for p in spec.extra_docker_compose if str(p).strip()]
    if extra:
        cfg["environment"]["extra_docker_compose"] = [str(Path(p).resolve()) for p in extra]

    if spec.dataset_fallback and not spec.tasks:
        cfg["datasets"] = [{"name": spec.dataset}]
        return cfg

    tasks = []
    for path in list_task_dirs(spec.tasks_root, spec.tasks or None):
        if not path.is_dir():
            raise FileNotFoundError(f"missing task dir: {path}")
        tasks.append({"path": str(path)})
    if not tasks:
        raise FileNotFoundError(f"no tasks found under {spec.tasks_root}")
    cfg["tasks"] = tasks
    return cfg


def write_job_config(spec: HarborJobSpec, out: str | Path, *, also_json: str | Path | None = None) -> dict[str, Any]:
    cfg = build_config(spec)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_yaml(cfg), encoding="utf-8")
    if also_json is not None:
        Path(also_json).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg
