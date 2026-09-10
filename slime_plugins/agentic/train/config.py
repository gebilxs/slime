from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Topology:
    pods: tuple[str, ...]
    gpus_per_node: int
    def __post_init__(self):
        if not self.pods: raise ValueError("topology pods must be explicitly pinned")
        if self.gpus_per_node <= 0: raise ValueError("gpus_per_node must be positive")

@dataclass(frozen=True)
class AgenticConfig:
    harness: str
    env_backend: str = "mock"
    actor: Topology | None = None
    rollout: Topology | None = None
    concurrency: int = 1
    n_samples_per_prompt: int = 1
    oversample: float = 1.0
    sidecar_endpoints: tuple[str, ...] = field(default_factory=tuple)
    options: dict[str, Any] = field(default_factory=dict)
    @property
    def actor_pods(self): return list(self.actor.pods) if self.actor else []
    @property
    def rollout_pods(self): return list(self.rollout.pods) if self.rollout else []
    def as_dict(self):
        out = dict(self.options)
        out.update({
            "harness": self.harness,
            "env_backend": self.env_backend,
            "concurrency": self.concurrency,
            "n_samples_per_prompt": self.n_samples_per_prompt,
            "oversample": self.oversample,
            "sidecar_endpoints": list(self.sidecar_endpoints),
        })
        if self.actor:
            out["actor"] = {"pods": list(self.actor.pods), "gpus_per_node": self.actor.gpus_per_node}
        if self.rollout:
            out["rollout"] = {"pods": list(self.rollout.pods), "gpus_per_node": self.rollout.gpus_per_node}
        return out

def _topology(name, value):
    if not isinstance(value, dict): raise ValueError(f"{name} topology must define pods and gpus_per_node")
    pods = value.get("pods"); pods = [pods] if isinstance(pods, str) else pods
    return Topology(tuple(pods or ()), int(value.get("gpus_per_node", 0)))

def resolve_config(data: dict[str, Any]) -> AgenticConfig:
    data = dict(data or {}); actor = _topology("actor", data.get("actor")); rollout = _topology("rollout", data.get("rollout"))
    env, ro = data.get("env") or {}, data.get("rollout") or {}
    sidecars = data.get("sidecar_endpoints", env.get("sidecar_endpoints", ())); sidecars = [sidecars] if isinstance(sidecars, str) else sidecars
    concurrency = int(data.get("concurrency", ro.get("concurrency", 1)))
    if concurrency <= 0: raise ValueError("concurrency must be positive")
    oversample = float(data.get("oversample", ro.get("oversample", 1.0)))
    if oversample < 1: raise ValueError("oversample must be >= 1")
    known = {"harness", "env", "actor", "rollout", "concurrency", "oversample", "n_samples_per_prompt", "sidecar_endpoints"}
    return AgenticConfig(str(data.get("harness", "")), str(env.get("backend", "mock")), actor, rollout, concurrency,
                         int(data.get("n_samples_per_prompt", 1)), oversample, tuple(sidecars or ()),
                         {k: v for k, v in data.items() if k not in known})
