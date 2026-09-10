"""CLI: python -m slime_plugins.agentic.harbor_terminus.prepare"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .images import prepare_tasks

_DEFAULT_TASKS = Path(
    "/mnt/shared-storage-gpfs2/trustcyberdata/private/xuchuokun/"
    "0805_ICLR_Submit/data/evals/terminal-bench-2"
)
_DEFAULT_MANIFEST = Path(
    "/mnt/shared-storage-gpfs2/trustcyberdata/private/xuchuokun/"
    "0805_ICLR_Submit/data/manifests/eval_tb2_tblite_images.jsonl"
)
_DEFAULT_TAR = Path("/mnt/shared-storage-gpfs2/trustcyberdata/private/image/h_eval_tb2")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load/tag TB2 images for Harbor Terminus")
    ap.add_argument("--tasks-root", type=Path, default=_DEFAULT_TASKS)
    ap.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    ap.add_argument("--tar-root", type=Path, default=_DEFAULT_TAR)
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--bake-tmux", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    summary = prepare_tasks(
        args.tasks_root,
        manifest=args.manifest if args.manifest.is_file() else None,
        tar_root=args.tar_root if args.tar_root.is_dir() else None,
        tasks=list(args.task) or None,
        dry_run=args.dry_run,
        pull=args.pull,
        bake=args.bake_tmux,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    print(f"done ok={summary['ok']}/{summary['n']}", flush=True)
    for rec in summary["rows"]:
        print(
            f"  {rec.get('task_id')} status={rec.get('status')} "
            f"expected={rec.get('expected')} tmux={rec.get('tmux', '-')}",
            flush=True,
        )
    return 0 if summary["ok"] == summary["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
