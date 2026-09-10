"""CLI: python -m slime_plugins.agentic.harbor_terminus — write a Harbor JobConfig YAML."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .job_config import (
    HarborJobSpec,
    resolve_harness,
    write_job_config,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write Harbor Terminus-2 job YAML")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks-root", type=Path, required=True)
    ap.add_argument("--jobs-dir", type=Path, required=True)
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument(
        "--harness",
        default="",
        help="evalkit.harness registry key: echo_xml | terminus-2 | oracle. "
        "Carries its own max_turns default and score lane. Preferred over --agent.",
    )
    ap.add_argument("--agent", default="terminus-2")
    ap.add_argument(
        "--agent-import-path",
        default="",
        help="Custom BaseAgent import path (module.path:ClassName or dotted). "
        "Wins over --agent when set.",
    )
    ap.add_argument("--model", default="openai/Qwen3-8B")
    ap.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--temperature", type=float, default=0.6)
    # None -> the harness registry supplies the default.
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--n-attempts", type=int, default=1)
    ap.add_argument("--n-concurrent", type=int, default=4)
    ap.add_argument(
        "--parser-name",
        default=None,
        help="Terminus-2 parser: json (Harbor default) or xml. "
        "Falls back to PARSER_NAME env, then xml.",
    )
    ap.add_argument(
        "--context-length",
        type=int,
        default=int(os.environ.get("CONTEXT_LEN") or 32768),
        help="Server context window. max_input + max_output must fit inside it.",
    )
    ap.add_argument("--max-input-tokens", type=int, default=None)
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--extra-docker-compose", action="append", default=[])
    ap.add_argument("--also-json", type=Path, default=None)
    ap.add_argument("--dataset-fallback", action="store_true")
    args = ap.parse_args(argv)

    parser_name = args.parser_name or os.environ.get("PARSER_NAME") or "xml"

    # Default the input budget to whatever the window leaves after the output
    # budget, so raising CONTEXT_LEN cannot silently overcommit.
    max_output = args.max_output_tokens if args.max_output_tokens is not None else 8192
    max_input = (
        args.max_input_tokens
        if args.max_input_tokens is not None
        else max(args.context_length - max_output, 1024)
    )

    spec = HarborJobSpec(
        job_name=args.job_name,
        jobs_dir=str(args.jobs_dir),
        tasks_root=str(args.tasks_root),
        n_attempts=args.n_attempts,
        n_concurrent=args.n_concurrent,
        harness=args.harness,
        agent=args.agent,
        agent_import_path=args.agent_import_path,
        model=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        max_turns=args.max_turns,
        parser_name=parser_name,
        context_length=args.context_length,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        extra_docker_compose=list(args.extra_docker_compose),
        tasks=list(args.task),
        dataset_fallback=bool(args.dataset_fallback),
    )
    try:
        cfg = write_job_config(spec, args.out, also_json=args.also_json)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    n = len(cfg.get("tasks") or cfg.get("datasets") or [])
    print(
        f"wrote {args.out} entries={n} n_attempts={spec.n_attempts} "
        f"harness={resolve_harness(spec).name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
