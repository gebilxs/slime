"""Train/eval protocol parity for the ECHO XML lane.

A train-matched eval is only meaningful if the eval drives the policy the way
training did. Nothing enforced that, and two kinds of drift were live:

  value drift    otendless_grpo_v1 trained at max_context_len=16384 while the
                 eval configs said 32768.

  default drift  recipes/train/load_train_yaml.py and
                 recipes/eval/load_eval_yaml.py disagree on the default for
                 the same YAML key: max_turns 6 vs 8, stop_on_action 0 vs 1,
                 max_consecutive_parse_fail 0 vs 2. Omit the key on one side
                 and the protocol changes silently.

This module resolves each side through its *own* loader defaults, so it
reports both kinds. Fields that may legitimately differ have to be named in
the eval config's `parity.allow_diff`, which makes every exception reviewable.

    python3 -m slime_plugins.agentic.protocols.echo_xml.protocol parity \
        --train configs/train/otendless_grpo_v1.yaml \
        --eval  configs/eval/echo_1node_oev1_i200_tblite.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Defaults as each loader actually applies them. Keep in sync with
# recipes/{train,eval}/load_*_yaml.py -- test_protocol_parity pins them.
TRAIN_DEFAULTS: dict[str, Any] = {
    "max_turns": 6,
    "max_context_len": 32768,
    "stop_on_action": 0,
    "max_new_tokens_per_turn": 4096,
    "user_role_obs": 0,
    "repeat_hint": 0,
    "episode_budget_s": 0,
    "gen_length_continue": 0,
    "few_shot": 0,
    "enable_thinking": 1,
    "no_think": 0,
    "paper_format": 0,
    "max_consecutive_parse_fail": 0,
    "max_obs_chars": 8000,
    "run_verifier": 1,
    "reward_mode": "oracle",
    "partial_reward": 0,
    "partial_bonus": 0.2,
    "format_coeff": 0,
}

EVAL_DEFAULTS: dict[str, Any] = {
    **TRAIN_DEFAULTS,
    "max_turns": 8,
    "stop_on_action": 1,
    "max_consecutive_parse_fail": 2,
    # load_eval_yaml.py now emits MAX_OBS_CHARS with this same 8000
    # default, so old yamls still resolve to the runtime-env value.
}

# How the model is prompted, stopped and truncated. Any difference here means
# the eval is not measuring the trained policy.
POLICY_FIELDS = (
    "max_turns",
    "max_context_len",
    "stop_on_action",
    "max_new_tokens_per_turn",
    "user_role_obs",
    "repeat_hint",
    "gen_length_continue",
    "few_shot",
    "enable_thinking",
    "no_think",
    "paper_format",
    "max_consecutive_parse_fail",
    "max_obs_chars",
    "episode_budget_s",
)

# How the score is produced. Differences change the number, not the policy.
SCORING_FIELDS = (
    "reward_mode",
    "run_verifier",
    "partial_reward",
    "partial_bonus",
    "format_coeff",
)

_FLOAT_FIELDS = frozenset({"episode_budget_s", "partial_bonus", "format_coeff"})
_STR_FIELDS = frozenset({"reward_mode"})


@dataclass(frozen=True)
class Resolved:
    value: Any
    explicit: bool


@dataclass(frozen=True)
class Mismatch:
    field: str
    train: Any
    eval: Any
    kind: str  # "value" | "default_drift"
    group: str  # "policy" | "scoring"

    def __str__(self) -> str:
        suffix = " (both sides relied on differing loader defaults)" if self.kind == "default_drift" else ""
        return f"[{self.group}] {self.field}: train={self.train!r} eval={self.eval!r}{suffix}"


def _coerce(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field in _STR_FIELDS:
        return str(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def protocol_block(cfg: dict) -> dict:
    return (cfg or {}).get("protocol") or {}


def resolve(block: dict, defaults: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Resolved]:
    out: dict[str, Resolved] = {}
    for field in fields:
        explicit = field in block
        raw = block.get(field, defaults.get(field))
        out[field] = Resolved(_coerce(field, raw), explicit)
    return out


def load_yaml(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"not a mapping: {path}")
    return cfg


def allow_list(eval_cfg: dict) -> tuple[str, ...]:
    parity = (eval_cfg or {}).get("parity") or {}
    return tuple(str(x) for x in (parity.get("allow_diff") or []))


def train_config_ref(eval_cfg: dict) -> str:
    parity = (eval_cfg or {}).get("parity") or {}
    return str(parity.get("train_config") or "")


def compare(
    train_cfg: dict,
    eval_cfg: dict,
    *,
    allow: tuple[str, ...] = (),
) -> list[Mismatch]:
    """Every policy/scoring field that differs and is not explicitly allowed."""
    fields = POLICY_FIELDS + SCORING_FIELDS
    tb, eb = protocol_block(train_cfg), protocol_block(eval_cfg)
    tr = resolve(tb, TRAIN_DEFAULTS, fields)
    er = resolve(eb, EVAL_DEFAULTS, fields)

    out: list[Mismatch] = []
    for field in fields:
        if field in allow:
            continue
        t, e = tr[field], er[field]
        if t.value == e.value:
            continue
        kind = "value" if (t.explicit or e.explicit) else "default_drift"
        group = "policy" if field in POLICY_FIELDS else "scoring"
        out.append(Mismatch(field, t.value, e.value, kind, group))
    return out


def report(
    train_path: str | Path,
    eval_path: str | Path,
    *,
    allow: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    train_cfg, eval_cfg = load_yaml(train_path), load_yaml(eval_path)
    allowed = tuple(allow) if allow is not None else allow_list(eval_cfg)
    mismatches = compare(train_cfg, eval_cfg, allow=allowed)
    fields = POLICY_FIELDS + SCORING_FIELDS
    return {
        "train_config": str(train_path),
        "eval_config": str(eval_path),
        "allow_diff": list(allowed),
        "ok": not mismatches,
        "mismatches": [
            {"field": m.field, "train": m.train, "eval": m.eval, "kind": m.kind, "group": m.group}
            for m in mismatches
        ],
        "resolved": {
            "train": {k: v.value for k, v in resolve(protocol_block(train_cfg), TRAIN_DEFAULTS, fields).items()},
            "eval": {k: v.value for k, v in resolve(protocol_block(eval_cfg), EVAL_DEFAULTS, fields).items()},
        },
    }


def to_harbor_kwargs(block: dict, *, side: str = "eval") -> dict[str, Any]:
    """EchoXmlAgent kwargs for a protocol block (see harbor_agent.__init__)."""
    defaults = TRAIN_DEFAULTS if side == "train" else EVAL_DEFAULTS
    r = resolve(block, defaults, POLICY_FIELDS)
    return {
        "max_turns": r["max_turns"].value,
        "max_obs_chars": r["max_obs_chars"].value,
        "user_role_obs": r["user_role_obs"].value,
        "enable_thinking": r["enable_thinking"].value,
        "episode_budget_s": r["episode_budget_s"].value,
        "max_new_tokens": r["max_new_tokens_per_turn"].value,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity", help="compare a train and an eval config")
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--eval", dest="eval_path", type=Path, required=True)
    p.add_argument("--allow", default=None, help="comma-separated fields permitted to differ")
    p.add_argument("--out", type=Path, default=None, help="write protocol_parity.json here")
    p.add_argument(
        "--warn-only",
        action="store_true",
        help="report and exit 0 (for lanes that are deliberately not train-matched)",
    )
    args = ap.parse_args(argv)

    try:
        eval_cfg = load_yaml(args.eval_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    train_path = args.train or train_config_ref(eval_cfg)
    if not train_path:
        print(
            f"ERROR: {args.eval_path} has no parity.train_config and --train was not given. "
            "A train-matched eval must name the run it claims to match.",
            file=sys.stderr,
        )
        return 2

    allow = tuple(x.strip() for x in args.allow.split(",") if x.strip()) if args.allow else None
    try:
        result = report(train_path, args.eval_path, allow=allow)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if result["ok"]:
        print(f"protocol parity ok: {args.eval_path} matches {train_path}")
        return 0

    print(f"protocol parity FAILED: {args.eval_path} vs {train_path}", file=sys.stderr)
    for m in result["mismatches"]:
        suffix = " (differing loader defaults)" if m["kind"] == "default_drift" else ""
        print(f"  - [{m['group']}] {m['field']}: train={m['train']!r} eval={m['eval']!r}{suffix}", file=sys.stderr)
    print(
        "  Fix the eval config, or list the field in parity.allow_diff with a comment "
        "explaining why the difference is legitimate.",
        file=sys.stderr,
    )
    return 0 if args.warn_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
