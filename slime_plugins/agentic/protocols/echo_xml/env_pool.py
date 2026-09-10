"""Back-compat shim. Moved to ``core.env_pool`` (harness-neutral).

The semaphore and executor are process-global and must stay shared, so this
re-exports the same objects rather than defining new ones.
"""

from __future__ import annotations

from core.env_pool import _docker_sem, _env_executor, env_timed, env_to_thread

__all__ = ["_docker_sem", "_env_executor", "env_timed", "env_to_thread"]
