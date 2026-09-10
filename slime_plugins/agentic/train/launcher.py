"""Resolve an agentic YAML. Does not submit Ray jobs.

Prints the resolved config and the live ``core.harness`` slime flags.
Cluster bringup and ``recipes/train/from_yaml.sh`` still own submission.
"""

from __future__ import annotations

import argparse

import yaml

from core.harness import train_hooks

from .config import resolve_config


def load_config(path):
    with open(path) as f:
        return resolve_config(yaml.safe_load(f) or {})


def slime_flags(cfg):
    return train_hooks(cfg.harness).slime_flags()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Resolve an agentic config. Use recipes/train/from_yaml.sh to submit."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--print-resolved", action="store_true", default=True)
    p.add_argument("--print-flags", action="store_true", default=True)
    ns = p.parse_args(argv)
    cfg = load_config(ns.config)
    if ns.print_resolved:
        print(yaml.safe_dump(cfg.as_dict(), sort_keys=False))
    if ns.print_flags:
        print(" ".join(slime_flags(cfg)))
    return cfg


if __name__ == "__main__":
    main()
