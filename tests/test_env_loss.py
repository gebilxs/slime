"""CPU unit tests for slime.utils.env_loss. No GPU / Megatron required."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from slime.utils.env_loss import (
    ENV_METRIC_KEYS,
    EnvLossConfig,
    apply_env_loss,
    compute_env_loss,
    compute_world_model_loss,
    env_coeff_from_args,
)

NUM_GPUS = 0


def _example_tensors():
    # Two assistant tokens, two observation tokens. |O| = 4 for the paper denom.
    log_probs = torch.tensor([-1.0, -2.0, -0.5, -0.5], dtype=torch.float32)
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    foc = torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch.float32)
    return log_probs, mask, foc


@pytest.mark.unit
def test_env_loss_scales_with_coeff():
    log_probs, mask, foc = _example_tensors()
    loss_a, metrics = compute_env_loss(log_probs, mask, coeff=0.05, full_observation_count=foc)
    loss_b, _ = compute_env_loss(log_probs, mask, coeff=0.1, full_observation_count=foc)
    assert loss_a is not None and loss_b is not None
    torch.testing.assert_close(loss_b, loss_a * 2)
    assert metrics["env/world_tokens_selected"].item() == 2


@pytest.mark.unit
def test_env_loss_disabled_when_coeff_zero():
    log_probs, mask, _ = _example_tensors()
    loss, metrics = compute_env_loss(log_probs, mask, coeff=0.0)
    assert loss is None
    assert metrics == {}


@pytest.mark.unit
def test_env_loss_empty_mask_returns_zero_connected_to_graph():
    log_probs = torch.tensor([-1.0, -2.0], dtype=torch.float32, requires_grad=True)
    mask = torch.zeros(2)
    loss, metrics = compute_env_loss(log_probs, mask, coeff=0.05)
    assert loss is not None
    assert loss.item() == 0.0
    assert metrics["env/world_tokens_selected"].item() == 0
    assert tuple(metrics) == ENV_METRIC_KEYS
    loss.backward()
    assert log_probs.grad is not None


@pytest.mark.unit
def test_env_loss_metric_keys_match_whether_obs_tokens_present():
    """Microbatches with/without observation tokens must log the same keys.

    reduce_train_step_metrics sums per-mb value tensors; 12 vs 13 crashes the
    first train step (ot_endless XML, 2026-08-13).
    """
    log_probs, mask, foc = _example_tensors()
    _, with_obs = compute_env_loss(log_probs, mask, coeff=0.05, full_observation_count=foc)
    _, without_obs = compute_env_loss(log_probs, torch.zeros_like(mask), coeff=0.05)
    assert tuple(with_obs) == tuple(without_obs) == ENV_METRIC_KEYS


@pytest.mark.unit
def test_env_loss_mask_length_mismatch_raises():
    log_probs = torch.zeros(4)
    mask = torch.ones(3)
    with pytest.raises(ValueError, match="world_loss_mask length"):
        compute_env_loss(log_probs, mask, coeff=0.05)


@pytest.mark.unit
def test_full_observation_tokens_matches_hand_derived_value():
    log_probs, mask, foc = _example_tensors()
    # world_ce on selected tokens = 0.5, 0.5; each divided by |O|=4; sum = 0.25
    # λ = 0.05 → 0.0125
    loss, metrics = compute_env_loss(
        log_probs,
        mask,
        coeff=0.05,
        full_observation_count=foc,
        normalization="full_observation_tokens",
    )
    torch.testing.assert_close(loss, torch.tensor(0.0125))
    torch.testing.assert_close(metrics["env/world_loss_unscaled"], torch.tensor(0.25))
    torch.testing.assert_close(metrics["env/world_ce_selected_per_token"], torch.tensor(0.5))


@pytest.mark.unit
def test_token_mean_normalization():
    log_probs, mask, _ = _example_tensors()
    # mean CE on selected tokens = (0.5 + 0.5) / 2 = 0.5; λ = 0.05 → 0.025
    loss, metrics = compute_env_loss(log_probs, mask, coeff=0.05, normalization="token_mean")
    torch.testing.assert_close(loss, torch.tensor(0.025))
    torch.testing.assert_close(metrics["env/world_loss_unscaled"], torch.tensor(0.5))


@pytest.mark.unit
def test_unknown_normalization_raises():
    log_probs, mask, _ = _example_tensors()
    with pytest.raises(ValueError, match="Unknown env-loss normalization"):
        compute_env_loss(log_probs, mask, coeff=0.05, normalization="not_a_mode")


@pytest.mark.unit
def test_world_model_wrapper_uses_config():
    log_probs, mask, foc = _example_tensors()
    cfg = EnvLossConfig(world_model_coeff=0.05, world_loss_normalization="full_observation_tokens")
    loss, _ = compute_world_model_loss(log_probs, mask, cfg, full_observation_count=foc)
    torch.testing.assert_close(loss, torch.tensor(0.0125))


@pytest.mark.unit
def test_env_coeff_from_args():
    assert env_coeff_from_args(SimpleNamespace(env_coeff=0.05, echo_coeff=0.1)) == 0.05
    assert env_coeff_from_args(SimpleNamespace(echo_coeff=0.1)) == 0.1
    assert env_coeff_from_args(SimpleNamespace()) == 0.0


@pytest.mark.unit
def test_apply_env_loss_adds_term():
    log_probs, mask, foc = _example_tensors()
    loss = torch.tensor(1.0)
    batch = {"world_loss_masks": [mask], "echo_full_obs_counts": [foc]}
    args = SimpleNamespace(env_coeff=0.05, echo_loss_normalization="full_observation_tokens")
    new_loss, metrics = apply_env_loss(loss, log_probs, batch, args)
    torch.testing.assert_close(new_loss, torch.tensor(1.0125))
    torch.testing.assert_close(metrics["env/world_loss_scaled"], torch.tensor(0.0125))


@pytest.mark.unit
def test_apply_env_loss_noop_when_disabled():
    log_probs, mask, _ = _example_tensors()
    loss = torch.tensor(1.0)
    new_loss, metrics = apply_env_loss(
        loss, log_probs, {"world_loss_masks": [mask]}, SimpleNamespace(env_coeff=0.0)
    )
    assert new_loss is loss
    assert metrics == {}
    new_loss, metrics = apply_env_loss(loss, log_probs, {}, SimpleNamespace(env_coeff=0.05))
    assert new_loss is loss
    assert tuple(metrics) == ENV_METRIC_KEYS
    torch.testing.assert_close(metrics["env/world_loss_scaled"], torch.tensor(0.0))


@pytest.mark.unit
def test_apply_env_loss_slice_cp():
    def slice_cp(tensor, _total_length, _response_length):
        return tensor[-2:]

    log_probs = torch.tensor([-0.5, -0.5], dtype=torch.float32)
    loss = torch.tensor(0.0)
    batch = {
        "world_loss_masks": [torch.tensor([0.0, 0.0, 1.0, 1.0])],
        "echo_full_obs_counts": [torch.tensor([4.0, 4.0, 4.0, 4.0])],
    }
    args = SimpleNamespace(env_coeff=0.05, echo_loss_normalization="full_observation_tokens")
    new_loss, _ = apply_env_loss(
        loss,
        log_probs,
        batch,
        args,
        slice_cp=slice_cp,
        total_lengths=[8],
        response_lengths=[4],
    )
    torch.testing.assert_close(new_loss, torch.tensor(0.0125))


@pytest.mark.unit
def test_apply_env_loss_slice_cp_requires_lengths():
    with pytest.raises(ValueError, match="slice_cp requires"):
        apply_env_loss(
            torch.tensor(0.0),
            torch.tensor([-0.5]),
            {"world_loss_masks": [torch.tensor([1.0])]},
            SimpleNamespace(env_coeff=0.05),
            slice_cp=lambda tensor, *_: tensor,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
