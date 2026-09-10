"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.

Rejection sampling (off by default): with ``--fa-reject-zero-std`` /
``SLIME_FA_REJECT_ZERO_STD=1`` the collection loop drops groups whose
in-group reward std is zero (all-correct / all-wrong, GRPO-degenerate) and
lets the worker's top-up pull replacement prompts from ``data_buffer`` until
``rollout_batch_size`` usable groups are collected. The drop budget per
rollout is ``ceil(rollout_batch_size * fa_reject_oversample) -
rollout_batch_size``; once it is exhausted, degenerate groups are accepted
again so a rollout always terminates (same fallback idea as
``check_reward_nonzero_std_with_fallback``). Filtering lives at collection
time, not in the done-callback, so the ABORTED-requeue and queue
backpressure semantics above are untouched.

Overlong drop (always on when ``args.seq_length`` is set): groups containing
a sample whose total token length exceeds the train-side ``--seq-length``
cap are dropped at collection time, unconditionally (no budget — an overlong
sample can never be trained, so there is no accept-anyway fallback). The
drop is group-atomic: per-sample drops would break the
``rollout_batch_size * n_samples_per_prompt`` sample count that GRPO reward
normalization and the DP/mbs scheduler rely on. Counts and the sample-level
rate ship per rollout as ``mini/reject_overlong_*`` in
``RolloutFnTrainOutput.metrics``.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import math
import os
import queue
import threading
import time

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _server_concurrency(args) -> int:
    """Per-engine in-flight group budget.

    CLI and ``SGLANG_SERVER_CONCURRENCY`` must agree when both are set. The
    previous launcher passed only the env var; argparse then used default 512
    and this file ignored the env, prefetching 512 * n_engines groups.
    """
    cli = getattr(args, "sglang_server_concurrency", None)
    env_raw = os.environ.get("SGLANG_SERVER_CONCURRENCY")
    env = int(env_raw) if env_raw not in (None, "") else None
    if cli is not None and env is not None and int(cli) != env:
        raise RuntimeError(
            f"sglang_server_concurrency mismatch: CLI={cli} env={env}. "
            "Pass --sglang-server-concurrency and keep the env in sync."
        )
    if cli is not None:
        val = int(cli)
    elif env is not None:
        val = env
    else:
        val = 8
    if val >= 512:
        logger.warning(
            "fully-async per-engine concurrency=%s looks like the argparse "
            "default; expected a small YAML value such as 4",
            val,
        )
    return max(1, val)


def _reject_zero_std_enabled(args) -> bool:
    """Collection-time zero-variance group rejection switch.

    CLI (``--fa-reject-zero-std``) and env (``SLIME_FA_REJECT_ZERO_STD``)
    both only turn the feature on: ``store_true`` cannot express an explicit
    CLI "off", so unlike :func:`_server_concurrency` there is no agreement
    check — either source enables it.
    """
    cli = bool(getattr(args, "fa_reject_zero_std", False))
    env_raw = os.environ.get("SLIME_FA_REJECT_ZERO_STD")
    env = bool(env_raw) and env_raw.strip().lower() in ("1", "true", "yes", "on")
    return cli or env


def _reject_oversample(args) -> float:
    """Oversample factor bounding the per-rollout drop budget.

    CLI and ``SLIME_FA_REJECT_OVERSAMPLE`` must agree when both are set
    (same fail-loud contract as :func:`_server_concurrency`).
    """
    cli = getattr(args, "fa_reject_oversample", None)
    env_raw = os.environ.get("SLIME_FA_REJECT_OVERSAMPLE")
    env = float(env_raw) if env_raw not in (None, "") else None
    if cli is not None and env is not None and float(cli) != env:
        raise RuntimeError(
            f"fa_reject_oversample mismatch: CLI={cli} env={env}. "
            "Pass --fa-reject-oversample and keep the env in sync."
        )
    val = float(cli) if cli is not None else (env if env is not None else 2.0)
    if val < 1.0:
        raise ValueError(f"fa_reject_oversample must be >= 1.0, got {val}")
    return val


def _zero_std_group(args, group: list[Sample]) -> bool:
    """True when the group carries no reward signal for GRPO (all-equal
    rewards). Mirrors
    ``filter_hub.dynamic_sampling_filters.check_reward_nonzero_std``
    (unbiased std, 1e-6 threshold) in pure Python so this module keeps its
    no-torch import surface. Groups shorter than 2 samples have no variance
    signal and are always kept.
    """
    if len(group) < 2:
        return False
    rewards = [float(sample.get_reward_value(args)) for sample in group]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    return math.sqrt(var) <= 1e-6


def _max_train_len(args) -> int:
    """Train-side sequence cap: samples longer than ``args.seq_length`` cannot
    be trained (the DP scheduler parks an over-cap sample in its own mbs, so a
    >seq_length sample silently runs an oversized logprob fwd+bwd and OOMs).
    0 disables the overlong drop (arg missing/unset)."""
    return int(getattr(args, "seq_length", 0) or 0)


def _overlong_group(group: list[Sample], max_train_len: int) -> bool:
    """True when any sample's total token length exceeds the train-side cap.

    The drop is group-atomic on purpose: per-sample drops would break the
    ``rollout_batch_size * n_samples_per_prompt`` sample count, which (a)
    flips GRPO reward normalization from per-group to the global fallback in
    ``RolloutManager._post_process_rewards`` and (b) can starve
    ``build_dp_schedule`` below one ``global_batch_size`` step (hard assert).
    Dropping the whole group keeps group sizes uniform and lets the worker
    top-up pull a replacement prompt, same as the zero-std reject.
    """
    return any(len(sample.tokens or []) > max_train_len for sample in group)


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            per_engine = _server_concurrency(args)
            n_engines = get_rollout_num_engines(args)
            total = per_engine * n_engines
            logger.info(
                "starting fully-async rollout worker per_engine=%s engines=%s "
                "inflight_prompt_groups=%s",
                per_engine,
                n_engines,
                total,
            )
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=total)
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        # Unbounded on purpose: put() runs inside the event-loop thread (task
        # done-callback), so a bounded queue that fills up would block the loop
        # and freeze every in-flight generation. Backpressure lives in _loop()
        # instead, which stops topping up while a full pool of completed groups
        # is already waiting to be consumed.
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue()
        self.poll_interval = 1.0
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        # Weight-update barrier: callers can stop intake, wait for active
        # generations to finish, requeue aborted groups, then resume.
        self.accepting = True
        self._active_count = 0
        self._active_condition = threading.Condition()

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self, limit: int | None = None) -> list[tuple[int, list[Sample]]]:
        """Pop up to ``limit`` completed groups (all of them when ``None``).

        Callers that only need a fixed number of groups must pass ``limit`` —
        anything popped beyond it would otherwise have to be thrown away, and
        these groups are fully generated and reward-scored, with their prompts
        already consumed from ``data_buffer``.
        """
        completed: list[tuple[int, list[Sample]]] = []
        while limit is None or len(completed) < limit:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def pause_for_weight_update(self, timeout: float = 30.0) -> bool:
        """Stop accepting prompts and wait for in-flight tasks to drain."""
        self.accepting = False
        deadline = time.time() + timeout
        with self._active_condition:
            while self._active_count and time.time() < deadline:
                self._active_condition.wait(timeout=max(0.01, deadline - time.time()))
            return self._active_count == 0

    def resume_after_weight_update(self) -> None:
        self.accepting = True

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        while self.running:
            try:
                # Reap done tasks
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()  # results already handled in callback
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done

                # Top up. The qsize gate is the queue's backpressure: once a
                # full pool of completed groups is waiting, stop pulling new
                # prompts until the training side drains some.
                while (
                    self.accepting
                    and len(active_tasks) < max_concurrent and self.output_queue.qsize() < max_concurrent and self.running
                ):
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        with self._active_condition:
                            self._active_count += 1
                        task.add_done_callback(self._make_done_cb(gid))
                        active_tasks.add(task)

                await asyncio.sleep(self.poll_interval)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(self.poll_interval)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            with self._active_condition:
                self._active_count = max(0, self._active_count - 1)
                self._active_condition.notify_all()
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return
            # Aborted group → requeue, don't ship to training.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    reject = _reject_zero_std_enabled(args)
    reject_budget = 0
    if reject:
        # DAPO-style oversampling budget: total groups burned per rollout is
        # capped at ceil(target * oversample); once the drop budget is gone we
        # accept degenerate groups again, so the rollout always terminates.
        reject_budget = max(0, math.ceil(target * _reject_oversample(args)) - target)
    max_train_len = _max_train_len(args)
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d inflight_budget=%s reject_zero_std=%s reject_budget=%d max_train_len=%d",
        rollout_id,
        target,
        worker.queue_size(),
        worker.concurrency,
        reject,
        reject_budget,
        max_train_len,
    )

    collected: dict[int, list[Sample]] = {}
    rejected = 0
    overlong_groups = 0
    overlong_samples = 0
    fallback_logged = False
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull only what this rollout still needs; the surplus stays queued for
        # the next rollout (that is the "queue stays warm" contract).
        drained = 0
        for gid, group in worker.get_completed_groups(limit=target - len(collected)):
            drained += 1
            if max_train_len > 0 and _overlong_group(group, max_train_len):
                # Unconditional (no budget): an overlong sample can never be
                # trained, so there is no accept-anyway fallback. The prompt is
                # consumed; the worker top-up pulls a replacement on its own.
                overlong_groups += 1
                overlong_samples += len(group)
                continue
            if reject and _zero_std_group(args, group):
                if rejected < reject_budget:
                    # Drop the degenerate group. The prompt is consumed; the
                    # freed queue slot re-opens the _loop top-up gate, which
                    # pulls a replacement prompt from data_buffer on its own.
                    rejected += 1
                    continue
                if not fallback_logged:
                    logger.warning(
                        "fully-async rollout %d: reject budget %d exhausted; "
                        "accepting zero-std groups again",
                        rollout_id,
                        reject_budget,
                    )
                    fallback_logged = True
            collected[gid] = group

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, rejected=%d, overlong_dropped=%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                rejected,
                overlong_groups,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)
    kept_samples = sum(len(group) for group in out)
    produced = kept_samples + overlong_samples
    metrics = {
        # Consumed by the mini_swe rollout-stats hook (rollout_extra_metrics).
        "mini/reject_overlong_rate": (overlong_samples / produced) if produced else 0.0,
        "mini/reject_overlong_groups": float(overlong_groups),
        "mini/reject_overlong_samples": float(overlong_samples),
    }
    logger.info(
        "fully-async rollout %d: done in %.1fs, rejected=%d, overlong_dropped=%d groups/%d samples, queue_left=%d",
        rollout_id,
        time.time() - started,
        rejected,
        overlong_groups,
        overlong_samples,
        worker.queue_size(),
    )
    return RolloutFnTrainOutput(samples=out, metrics=metrics)


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
