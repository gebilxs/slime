"""Small synchronization primitive used around rollout weight updates."""
from __future__ import annotations
import asyncio

class WeightSyncBarrier:
    """Stop intake, wait for in-flight work, then resume after sync.

    ``abort`` callback should return groups that must be requeued.  The class
    is backend agnostic and therefore straightforward to exercise in CPU
    contract tests.
    """
    def __init__(self):
        self.accepting = True
        self._inflight = set()
        self._condition = asyncio.Condition()

    async def acquire(self):
        async with self._condition:
            while not self.accepting:
                await self._condition.wait()
            token = object(); self._inflight.add(token); return token

    async def release(self, token):
        async with self._condition:
            self._inflight.discard(token); self._condition.notify_all()

    async def pause(self):
        async with self._condition:
            self.accepting = False
            while self._inflight:
                await self._condition.wait()

    async def resume(self):
        async with self._condition:
            self.accepting = True; self._condition.notify_all()

    async def synchronize(self, update):
        await self.pause()
        try:
            result = update()
            if asyncio.iscoroutine(result): result = await result
            return result
        finally:
            await self.resume()

__all__ = ["WeightSyncBarrier"]
