"""GPU semantic tests for slime.utils.env_loss.

Uses a tiny GRU language model on 1 GPU. Does not load Qwen, start Docker,
or touch Megatron. Random pretrained NLL is not the test: after a short fit
on real observation tokens, L_Env(real obs) must be lower than L_Env(fake obs).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from slime.utils.env_loss import compute_env_loss

NUM_GPUS = 1

VOCAB = 48
HIDDEN = 32
PROMPT_LEN = 4
ACTION_LEN = 3
OBS_LEN = 8


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB, hidden: int = HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.embed(input_ids))
        return self.lm_head(hidden)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def _response_log_probs_and_mask(
    model: TinyCausalLM,
    tokens: torch.Tensor,
    prompt_len: int = PROMPT_LEN,
    action_len: int = ACTION_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """log π(response_t | prompt + response_<t) and env mask on observation tokens."""
    logits = model(tokens)
    response = tokens[0, prompt_len:]
    pred_logits = logits[0, prompt_len - 1 : -1]
    log_probs = F.log_softmax(pred_logits, dim=-1).gather(-1, response.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(log_probs)
    mask[action_len:] = 1.0
    return log_probs, mask


@pytest.mark.gpu
def test_env_loss_matches_masked_ce_on_cuda():
    device = _require_cuda()
    torch.manual_seed(0)
    model = TinyCausalLM().to(device)
    tokens = torch.randint(0, VOCAB, (1, PROMPT_LEN + ACTION_LEN + OBS_LEN), device=device)
    log_probs, mask = _response_log_probs_and_mask(model, tokens)
    logits = model(tokens)
    obs_logits = logits[0, PROMPT_LEN + ACTION_LEN - 1 : -1]
    obs_targets = tokens[0, PROMPT_LEN + ACTION_LEN :]
    ce = F.cross_entropy(obs_logits, obs_targets, reduction="mean")
    loss, metrics = compute_env_loss(log_probs, mask, coeff=0.05, normalization="token_mean")
    assert loss is not None
    assert loss.device.type == "cuda"
    torch.testing.assert_close(loss, ce * 0.05)
    torch.testing.assert_close(metrics["env/world_tokens_selected"], mask.sum())


@pytest.mark.gpu
def test_env_loss_gradients_only_on_observation_log_probs():
    device = _require_cuda()
    torch.manual_seed(1)
    model = TinyCausalLM().to(device)
    tokens = torch.randint(0, VOCAB, (1, PROMPT_LEN + ACTION_LEN + OBS_LEN), device=device)
    log_probs, mask = _response_log_probs_and_mask(model, tokens)
    log_probs = log_probs.detach().requires_grad_(True)
    loss, _ = compute_env_loss(log_probs, mask, coeff=0.05, normalization="token_mean")
    assert loss is not None
    loss.backward()
    assert log_probs.grad is not None
    assert torch.all(log_probs.grad[:ACTION_LEN] == 0)
    assert log_probs.grad[ACTION_LEN:].abs().sum() > 0


@pytest.mark.gpu
def test_env_loss_lower_on_real_obs_than_fake_after_fit():
    device = _require_cuda()
    torch.manual_seed(2)
    model = TinyCausalLM().to(device)
    tokens = torch.randint(0, VOCAB, (1, PROMPT_LEN + ACTION_LEN + OBS_LEN), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(80):
        opt.zero_grad()
        log_probs, mask = _response_log_probs_and_mask(model, tokens)
        loss, _ = compute_env_loss(log_probs, mask, coeff=1.0, normalization="token_mean")
        assert loss is not None
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        log_real, mask = _response_log_probs_and_mask(model, tokens)
        loss_real, _ = compute_env_loss(log_real, mask, coeff=1.0, normalization="token_mean")
        fake = tokens.clone()
        fake[0, PROMPT_LEN + ACTION_LEN :] = (fake[0, PROMPT_LEN + ACTION_LEN :] + 7) % VOCAB
        log_fake, mask_fake = _response_log_probs_and_mask(model, fake)
        loss_fake, _ = compute_env_loss(log_fake, mask_fake, coeff=1.0, normalization="token_mean")

    assert loss_real is not None and loss_fake is not None
    assert loss_real.item() < loss_fake.item()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
