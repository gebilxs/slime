from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None

    # Backward-compatible sequence view for callers written before the
    # structured RolloutOutput contract. New code should use ``.samples``.
    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


# Generic name used by agentic plugins.  Keep the historical class as the
# concrete implementation so all existing callers and type checks continue to
# work while new code can depend on the protocol-neutral name.
RolloutOutput = RolloutFnTrainOutput


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        # Agentic plugins expose the generic RolloutOutput contract without
        # importing slime internals.  Normalize any object carrying the same
        # ``samples``/``metrics`` fields before falling back to legacy lists.
        if not evaluation and hasattr(output, "samples"):
            output = RolloutFnTrainOutput(
                samples=list(output.samples),
                metrics=dict(getattr(output, "metrics", None) or {}),
            )
        elif evaluation and hasattr(output, "data"):
            output = RolloutFnEvalOutput(
                data=dict(output.data),
                metrics=dict(getattr(output, "metrics", None) or {}),
            )
        else:
            output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
