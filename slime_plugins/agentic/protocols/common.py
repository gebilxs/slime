"""Mock-only episode helper. Not the training rollout.

Training uses ``slime_plugins.agentic.protocols.mini_swe.generate.run_mini_swe_episode`` and
``slime_plugins.agentic.protocols.echo_xml.generate.run_echo_episode``, which own chat templates,
per-turn SGLang logprobs, and loss masks. Do not feed ``ProtocolEpisode.result``
into GRPO.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from ..contracts import EpisodeResult, Transcript
from ..env.protocol import StepResult, TaskEnv


async def call_policy(policy: Callable[..., Any], observation: str, transcript: Transcript) -> str:
    try:
        value = policy(observation, transcript)
    except TypeError:
        value = policy(observation)
    if inspect.isawaitable(value):
        value = await value
    return "" if value is None else str(value)


class ProtocolEpisode:
    protocol = "generic"

    def __init__(self, env: TaskEnv, *, max_turns: int = 16, tokenizer: Any = None, metadata: dict[str, Any] | None = None):
        self.env = env
        self.max_turns = max_turns
        self.tokenizer = tokenizer
        self.metadata = dict(metadata or {})
        self.transcript: Transcript = []
        self.observation = ""
        self.turns = 0
        self.done = False
        self.total_reward = 0.0
        self.status = "running"

    async def reset(self) -> str:
        self.observation = await self.env.reset()
        self.transcript = [{"role": "system", "content": self.observation}]
        self.turns = 0
        self.done = False
        self.total_reward = 0.0
        self.status = "running"
        return self.observation

    async def step(self, action: str, truncated: bool = False) -> StepResult:
        if self.done:
            return StepResult(self.observation, done=True, reward=0.0, info={"already_done": True})
        result = await self.env.step(action, truncated=truncated)
        self.transcript.extend(
            (
                {"role": "assistant", "content": action},
                {"role": "tool", "content": result.observation},
            )
        )
        self.observation = result.observation
        self.turns += 1
        self.total_reward += float(result.reward or 0.0)
        self.done = bool(result.done or truncated or self.turns >= self.max_turns)
        if self.done:
            hit_cap = (truncated or self.turns >= self.max_turns) and not result.done
            self.status = "truncated" if hit_cap else "completed"
        return result

    async def run(self, policy: Callable[..., Any], *, initial_observation: str | None = None) -> EpisodeResult:
        try:
            obs = await self.reset() if initial_observation is None else initial_observation
            while not self.done and self.turns < self.max_turns:
                action = await call_policy(policy, obs, self.transcript)
                result = await self.step(action)
                obs = result.observation
            return self.result()
        finally:
            close = getattr(self.env, "close", None)
            if close is None:
                pass
            else:
                try:
                    value = close()
                    if inspect.isawaitable(value):
                        await value
                except Exception:
                    pass

    def result(self) -> EpisodeResult:
        """Mock transcript only. Token lanes stay empty so they cannot be trained."""
        response_text = "\n".join(
            m["content"] for m in self.transcript[1:] if m["role"] == "assistant"
        )
        return EpisodeResult(
            status=self.status,
            prompt_ids=[],
            response=response_text,
            response_token_ids=[],
            loss_mask=[],
            world_loss_mask=[],
            rollout_log_probs=[],
            metadata={**self.metadata, "turns": self.turns, "mock_episode": True},
            verifier_score=self.total_reward,
            full_obs_token_count=0,
        )
