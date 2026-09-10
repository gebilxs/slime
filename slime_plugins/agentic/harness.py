"""Plugin harness view of the live ``core.harness`` registry.

Train generate / reward / rollout-log flags point at this package. ``protocols.*``
is a compatibility shim that re-exports the same callables.
"""

from __future__ import annotations

from core.harness import (
    HARNESSES,
    KIND_BARE,
    KIND_BUILTIN,
    KIND_IMPORT,
    HarborAgent,
    HarnessSpec,
    TrainHooks,
    get,
    names,
    poolable,
    train_hooks,
    trainable_names,
)

PLUGIN_TRAIN = {
    "mini_swe": TrainHooks(
        generate="slime_plugins.agentic.protocols.mini_swe.generate.generate",
        reward="slime_plugins.agentic.protocols.mini_swe.reward.reward_func",
        rollout_log="slime_plugins.agentic.protocols.mini_swe.rollout_stats.log_rollout",
    ),
    "echo_xml": TrainHooks(
        generate="slime_plugins.agentic.protocols.echo_xml.generate.generate",
        reward="slime_plugins.agentic.protocols.echo_xml.reward.reward_func",
        rollout_log="slime_plugins.agentic.protocols.echo_xml.rollout_stats.log_rollout",
    ),
}


def plugin_train_hooks(name: str) -> TrainHooks:
    try:
        return PLUGIN_TRAIN[name]
    except KeyError:
        return train_hooks(name)


def register(name, harness):
    raise RuntimeError(
        "register harnesses in core.harness.HARNESSES; "
        f"this facade does not own a second registry ({name!r})"
    )


def get_harness(name):
    return get(name)


def register_harness(name):
    def deco(obj):
        register(name, obj)
        return obj

    return deco


__all__ = [
    "HARNESSES",
    "KIND_BARE",
    "KIND_BUILTIN",
    "KIND_IMPORT",
    "HarborAgent",
    "HarnessSpec",
    "PLUGIN_TRAIN",
    "TrainHooks",
    "get",
    "get_harness",
    "names",
    "plugin_train_hooks",
    "poolable",
    "register",
    "register_harness",
    "train_hooks",
    "trainable_names",
]
