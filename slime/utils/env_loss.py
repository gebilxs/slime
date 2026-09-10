"""Environment-prediction loss L_Env for agentic RL (ECHO, arXiv:2605.24517).

Paper: L = L_GRPO + λ L_Env on observation tokens.
Upstream: echo_rl.world_modeling.loss.compute_world_model_loss

This module is backend-agnostic: it consumes already-concatenated 1-D
response log-probs (slime's packed layout, length == response_length per
sample). Megatron / context-parallel alignment belongs in the backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Must be identical for every microbatch. A missing key here makes
# reduce_train_step_metrics crash (tensor 12 vs 13) when one mb has
# observation tokens and another does not.
ENV_METRIC_KEYS = (
    "env/world_loss_unscaled",
    "env/world_loss_scaled",
    "env/world_tokens_selected",
    "env/world_ce_selected_per_token",
    "env/coeff",
)


@dataclass
class EnvLossConfig:
    world_model_coeff: float = 0.05
    # "token_mean" — CE sum / selected tokens
    # "full_observation_tokens" — each selected token divided by |O| then summed (paper default)
    world_loss_normalization: str = "full_observation_tokens"


def _env_metrics(
    *,
    device,
    coeff: float,
    world_loss_unscaled: torch.Tensor,
    world_loss_scaled: torch.Tensor,
    world_tokens_selected: torch.Tensor,
    world_ce_selected_per_token: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "env/world_loss_unscaled": world_loss_unscaled.detach(),
        "env/world_loss_scaled": world_loss_scaled.detach(),
        "env/world_tokens_selected": world_tokens_selected.detach(),
        "env/world_ce_selected_per_token": world_ce_selected_per_token.detach(),
        "env/coeff": torch.tensor(float(coeff), device=device),
    }


def _zero_env_metrics(zero: torch.Tensor, coeff: float) -> dict[str, torch.Tensor]:
    z = (zero * 0.0).detach()
    return _env_metrics(
        device=zero.device,
        coeff=coeff,
        world_loss_unscaled=z,
        world_loss_scaled=z,
        world_tokens_selected=z,
        world_ce_selected_per_token=z,
    )


def compute_env_loss(
    log_probs: torch.Tensor,
    world_loss_mask: torch.Tensor,
    *,
    coeff: float,
    full_observation_count: torch.Tensor | None = None,
    normalization: str = "full_observation_tokens",
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Compute λ * L_Env on already-concatenated response log-probs.

    Args:
        log_probs: 1-D tensor, cat of per-sample response log-probs.
        world_loss_mask: 1-D float/int mask, same length as log_probs (1 = env CE target).
        coeff: λ (world_model_coeff). 0 disables.
        full_observation_count: 1-D float tensor same length as log_probs; each
            position holds that sample's |O| (total observation tokens). Used
            when normalization == "full_observation_tokens".
        normalization: see EnvLossConfig.
    """
    if coeff is None or coeff <= 0 or world_loss_mask is None:
        return None, {}

    device = log_probs.device
    mask = world_loss_mask.to(device=device, dtype=torch.float32)
    if mask.numel() != log_probs.numel():
        raise ValueError(f"world_loss_mask length {mask.numel()} != log_probs length {log_probs.numel()}")

    world_ce = -log_probs
    selected = mask.sum()
    zero = log_probs.sum() * 0.0

    if selected <= 0:
        metrics = _zero_env_metrics(zero, coeff)
        metrics["env/world_tokens_selected"] = selected.detach()
        return zero, metrics

    # selected > 0 is guaranteed above, so no epsilon on the denominator.
    per_token_ce = (world_ce * mask).sum() / selected
    if normalization == "full_observation_tokens":
        if full_observation_count is None:
            world_loss_unscaled = per_token_ce
        else:
            denom = full_observation_count.to(device=device, dtype=torch.float32)
            if denom.numel() != log_probs.numel():
                raise ValueError(
                    f"full_observation_count length {denom.numel()} != log_probs length {log_probs.numel()}"
                )
            world_loss_unscaled = ((world_ce * mask) / denom.clamp_min(1e-12)).sum()
    elif normalization == "token_mean":
        world_loss_unscaled = per_token_ce
    else:
        raise ValueError(f"Unknown env-loss normalization: {normalization}")

    world_loss_scaled = coeff * world_loss_unscaled
    return world_loss_scaled, _env_metrics(
        device=device,
        coeff=coeff,
        world_loss_unscaled=world_loss_unscaled,
        world_loss_scaled=world_loss_scaled,
        world_tokens_selected=selected,
        world_ce_selected_per_token=per_token_ce,
    )


def env_coeff_from_args(args) -> float:
    """λ from --env-coeff / --echo-coeff (0 disables)."""
    return float(getattr(args, "env_coeff", None) or getattr(args, "echo_coeff", 0.0) or 0.0)


def _as_1d_float(value, device=None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
    tensor = tensor.to(dtype=torch.float32).reshape(-1)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def apply_env_loss(
    loss: torch.Tensor,
    log_probs: torch.Tensor,
    batch: dict,
    args,
    *,
    slice_cp=None,
    total_lengths=None,
    response_lengths=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add λ L_Env to an already-reduced policy loss. No-op when coeff=0.

    When coeff>0 but masks are missing, returns zero metrics with the same
    keys as a real L_Env step so DP ranks / microbatches still reduce.

    ``slice_cp`` is the backend's context-parallel slice (Megatron
    ``slice_log_prob_with_cp``). Pass it only when CP>1; masks are stored at
    full response length and must match already-sliced ``log_probs``.
    """
    coeff = env_coeff_from_args(args)
    masks = batch.get("world_loss_masks") if batch else None
    if coeff <= 0:
        return loss, {}
    if not masks:
        return loss, _zero_env_metrics(loss, coeff)

    focs = batch.get("echo_full_obs_counts")
    if slice_cp is not None:
        if total_lengths is None or response_lengths is None:
            raise ValueError("slice_cp requires total_lengths and response_lengths")
        foc_list = list(focs) if focs else [None] * len(masks)
        sliced_masks = []
        sliced_focs = []
        for mask_i, foc_i, total_length, response_length in zip(
            masks, foc_list, total_lengths, response_lengths
        ):
            sliced_masks.append(slice_cp(_as_1d_float(mask_i), total_length, response_length))
            if foc_i is not None:
                sliced_focs.append(slice_cp(_as_1d_float(foc_i), total_length, response_length))
        masks = sliced_masks
        focs = sliced_focs or None

    device = log_probs.device
    world_mask = torch.cat([_as_1d_float(mask, device=device) for mask in masks], dim=0)
    full_obs = None
    if focs:
        full_obs = torch.cat([_as_1d_float(foc, device=device) for foc in focs], dim=0)

    env_term, metrics = compute_env_loss(
        log_probs,
        world_mask,
        coeff=coeff,
        full_observation_count=full_obs,
        normalization=getattr(args, "echo_loss_normalization", None)
        or getattr(args, "env_loss_normalization", "full_observation_tokens"),
    )
    if env_term is not None:
        loss = loss + env_term
    return loss, metrics


def compute_world_model_loss(
    action_log_probs: torch.Tensor,
    world_loss_mask: torch.Tensor | None,
    config: EnvLossConfig,
    full_observation_count: torch.Tensor | None = None,
    **_unused,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Paper-facing wrapper (echo-rl name)."""
    return compute_env_loss(
        action_log_probs,
        world_loss_mask,
        coeff=config.world_model_coeff,
        full_observation_count=full_observation_count,
        normalization=config.world_loss_normalization,
    )
