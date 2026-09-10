"""Turn-level GRPO advantage for ECHO XML episodes (rllm loop style, no critic).

Math. Sample i of a same-prompt group has scalar episode reward r_i, turn token
spans ``turn_token_spans[i][t] = [start, end)`` (response-token space, t =
0..T_i-1) and response loss mask. The per-token advantage is

    A_i        = (r_i - mean_g(r)) / (std_g(r) + eps)   # group centering,
                                                        # identical to slime's
                                                        # _post_process_rewards
    A_{i,t}    = A_i * gamma^(T_i - 1 - t)              # per-turn discount:
                                                        # the last turn carries
                                                        # the most credit
    adv[i][p]  = A_{i,t(p)}   if loss_mask[i][p] == 1 and p in turn t's span
               = A_i          if loss_mask[i][p] == 1 but p is in no span
                                                        # (degenerate metadata;
                                                        # matches the scalar
                                                        # baseline for that token)
               = 0            otherwise                 # inert: the loss
                                                        # multiplies by loss_mask

Why this is "GAE without a critic": treat each turn as one macro step with
reward only at the terminal turn and a value function V(.) == 0. Then
GAE(gamma, lambda) gives

    A_t = sum_{l>=0} (gamma*lambda)^l delta_{t+l},
    delta_t = r_t + gamma*V(s_{t+1}) - V(s_t) = r_t,
    r_t = 0 for t < T-1, r_{T-1} = A_i
        =>  A_t = (gamma*lambda)^(T-1-t) * A_i.

So with no critic the lambda=0.95 GAE form degenerates to plain discounting of
the terminal (group-centered) advantage, and a single effective per-turn
discount gamma plays the role of the combined (gamma*lambda) factor. Our
default gamma=0.99 (env TURN_ADV_GAMMA) is a milder decay than
gamma*lambda = 1.0*0.95; lambda is not a separate knob because it is
unidentifiable without a learned V.

Wiring (all gated by TURN_ADV, default 0 = exact baseline behavior):

    rollout side (RolloutManager._convert_samples_to_train_data hook):
        --custom-convert-samples-to-train-data-path \\
            slime_plugins.agentic.protocols.echo_xml.turn_adv.convert_samples_to_train_data
        Wraps the default conversion (same code path, incl. any
        --custom-reward-post-process-path) and adds
        train_data["turn_token_advantages"]: per-token floats, one list per
        sample, response length. The scalar "rewards" stay untouched, so logs
        and metrics are unchanged.
    trainer side (compute_advantages_and_returns hook):
        --custom-advantage-function-path \\
            slime_plugins.agentic.protocols.echo_xml.turn_adv.advantage_function
        Replaces get_grpo_returns' scalar broadcast with the per-token values
        (CP-sliced like rollout_log_probs). Falls back to the exact GRPO
        broadcast when TURN_ADV != 1 or the key is absent.
    plumbing: slime/ray/rollout.py carries "turn_token_advantages" through
        _split_train_data_by_dp (additive; absent key = no-op).

This module is importable without slime/torch/megatron (unit tests run on a
login node); slime/torch/megatron imports are lazy inside the hook functions.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Sequence

logger = logging.getLogger(__name__)

ENV_SWITCH = "TURN_ADV"
ENV_GAMMA = "TURN_ADV_GAMMA"
DEFAULT_GAMMA = 0.99
TRAIN_DATA_KEY = "turn_token_advantages"


def turn_adv_enabled() -> bool:
    from core.environ import getenv

    return getenv(ENV_SWITCH, "0") == "1"


def turn_adv_gamma() -> float:
    from core.environ import getenv

    return float(getenv(ENV_GAMMA, str(DEFAULT_GAMMA)) or DEFAULT_GAMMA)


# ---------------------------------------------------------------------------
# Pure math (no slime/torch deps)
# ---------------------------------------------------------------------------


def turn_discount_weights(num_turns: int, gamma: float = DEFAULT_GAMMA) -> list[float]:
    """w_t = gamma^(T-1-t) for t = 0..T-1; the last turn has weight 1."""
    return [gamma ** (num_turns - 1 - t) for t in range(num_turns)]


def group_grpo_advantages(
    rewards: Sequence[float],
    group_size: int,
    *,
    std_normalization: bool = True,
    eps: float = 1e-6,
) -> list[float]:
    """Group-centered GRPO advantages, mirroring slime _post_process_rewards.

    Consecutive chunks of ``group_size`` form same-prompt groups. std is the
    unbiased (ddof=1) estimator, matching torch.Tensor.std used by slime.
    For group_size == 1, std is 0 and every advantage collapses to 0; slime's
    arguments.py force-disables grpo_std_normalization in that configuration,
    so callers should do the same.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    out: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = [float(r) for r in rewards[start : start + group_size]]
        mean = sum(group) / len(group)
        centered = [r - mean for r in group]
        if std_normalization:
            if len(group) > 1:
                var = sum(x * x for x in centered) / (len(group) - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            centered = [x / (std + eps) for x in centered]
        out.extend(centered)
    return out


def expand_turn_advantages(
    turn_token_spans: Sequence[Sequence[int]] | None,
    loss_mask: Sequence[int],
    advantage: float,
    gamma: float = DEFAULT_GAMMA,
) -> list[float]:
    """Broadcast a group-centered scalar advantage to response tokens per turn.

    Positions with loss_mask == 0 get 0 (inert downstream). Trainable positions
    covered by turn t's span get ``advantage * gamma^(T-1-t)``; trainable
    positions not covered by any span (should not happen for spans emitted by
    run_echo_episode) get the undiscounted ``advantage``, i.e. the exact
    scalar-broadcast baseline value for that token. ``None``/empty spans fall
    back to the scalar broadcast on every response token, matching
    get_grpo_returns exactly (foreign samples in mixed batches have no spans).
    """
    n = len(loss_mask)
    if not turn_token_spans:
        return [float(advantage)] * n
    adv = [0.0] * n
    covered = bytearray(n)
    weights = turn_discount_weights(len(turn_token_spans), gamma)
    for t, span in enumerate(turn_token_spans):
        s, e = int(span[0]), int(span[1])
        s = max(0, min(s, n))
        e = max(s, min(e, n))
        w = weights[t]
        for pos in range(s, e):
            if loss_mask[pos]:
                adv[pos] = float(advantage) * w
                covered[pos] = 1
    for pos in range(n):
        if loss_mask[pos] and not covered[pos]:
            adv[pos] = float(advantage)
    return adv


def compute_group_turn_advantages(
    group_rewards: Sequence[float],
    group_turn_token_spans: Sequence[Sequence[Sequence[int]] | None],
    group_loss_masks: Sequence[Sequence[int]],
    *,
    gamma: float = DEFAULT_GAMMA,
    std_normalization: bool = True,
    eps: float = 1e-6,
) -> list[list[float]]:
    """One same-prompt group in, per-token advantages out (GRPO-native loop)."""
    n = len(group_rewards)
    if not (len(group_turn_token_spans) == len(group_loss_masks) == n):
        raise ValueError("group_rewards / spans / loss_masks length mismatch")
    advs = group_grpo_advantages(group_rewards, n, std_normalization=std_normalization, eps=eps)
    return [
        expand_turn_advantages(spans, mask, adv, gamma)
        for spans, mask, adv in zip(group_turn_token_spans, group_loss_masks, advs, strict=True)
    ]


# ---------------------------------------------------------------------------
# slime rollout-side hook (--custom-convert-samples-to-train-data-path)
# ---------------------------------------------------------------------------


def convert_samples_to_train_data(args: Any, samples: list) -> dict:
    """Default conversion + optional train_data["turn_token_advantages"].

    Always delegates to RolloutManager._convert_samples_to_train_data (with any
    user-set custom reward post-process preserved), so TURN_ADV=0 yields
    byte-identical training data. When enabled, the per-token advantages are
    derived from train_data["rewards"] — i.e. from exactly the per-sample
    scalar the baseline GRPO would broadcast — so group centering stays
    whatever the active configuration produced.
    """
    train_data = _default_convert_samples_to_train_data(args, samples)
    if not turn_adv_enabled():
        return train_data
    attach_turn_advantages(train_data, samples, gamma=turn_adv_gamma())
    return train_data


def _default_convert_samples_to_train_data(args: Any, samples: list) -> dict:
    from slime.ray.rollout import RolloutManager  # lazy: needs ray/sglang
    from slime.utils.misc import load_function

    class _DefaultConvertShim:
        custom_convert_samples_to_train_data_func = None
        _post_process_rewards = RolloutManager._post_process_rewards

        def __init__(self, args):
            self.args = args
            path = getattr(args, "custom_reward_post_process_path", None)
            self.custom_reward_post_process_func = load_function(path) if path else None

    return RolloutManager._convert_samples_to_train_data(_DefaultConvertShim(args), samples)


def attach_turn_advantages(train_data: dict, samples: list, *, gamma: float = DEFAULT_GAMMA) -> None:
    """Attach per-token advantages to a default-converted train_data, in place.

    Samples without turn_token_spans metadata get the plain scalar broadcast,
    so mixed batches degrade gracefully per sample.
    """
    rewards = train_data["rewards"]
    loss_masks = train_data["loss_masks"]
    advs: list[list[float]] = []
    n_with_spans = 0
    for i, sample in enumerate(samples):
        meta = getattr(sample, "metadata", None) or {}
        spans = meta.get("turn_token_spans")
        if spans:
            n_with_spans += 1
        advs.append(expand_turn_advantages(spans, loss_masks[i], float(rewards[i]), gamma))
    train_data[TRAIN_DATA_KEY] = advs
    logger.info(
        "turn_adv: attached %s for %d/%d samples (gamma=%s)",
        TRAIN_DATA_KEY,
        n_with_spans,
        len(samples),
        gamma,
    )


# ---------------------------------------------------------------------------
# slime trainer-side hook (--custom-advantage-function-path)
# ---------------------------------------------------------------------------


def advantage_function(args: Any, rollout_data: dict) -> None:
    """Set rollout_data["advantages"]/["returns"] from per-token turn advantages.

    Fallback (TURN_ADV off or key absent) is exactly the GRPO branch of
    compute_advantages_and_returns: get_grpo_returns scalar broadcast.
    """
    kl = rollout_data["kl"]
    adv_lists = rollout_data.get(TRAIN_DATA_KEY)
    if not turn_adv_enabled() or adv_lists is None:
        import torch  # lazy: trainer-only dep

        from slime.utils.ppo_utils import get_grpo_returns

        rewards = torch.tensor(rollout_data["rewards"], dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        rollout_data["returns"] = returns
        rollout_data["advantages"] = [r for r in returns]
        return

    import torch  # lazy: trainer-only dep

    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    advantages = []
    for i, k in enumerate(kl):
        full = torch.as_tensor(adv_lists[i], dtype=torch.float32)
        local = slice_log_prob_with_cp(full, total_lengths[i], response_lengths[i])
        local = local.to(device=k.device, dtype=torch.float32)
        assert local.shape == k.shape, (
            f"turn_token_advantages shape {tuple(local.shape)} != kl shape {tuple(k.shape)} "
            f"for sample {i} (response_length={response_lengths[i]})"
        )
        advantages.append(local)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [a for a in advantages]
