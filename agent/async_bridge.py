"""Bridge between Streamlit's synchronous script model and the async agent.

Streamlit re-runs the script on every interaction. Calling ``asyncio.run()`` per rerun creates a
new event loop each time, which breaks long-lived async resources (e.g. LangChain's cached
``httpx.AsyncClient`` used by the OpenAI client: "Event loop is closed"). Instead we keep ONE
event loop alive in a daemon thread for the whole process and hand coroutines to it, streaming
their results back to the script thread through a thread-safe queue so the UI updates live.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TypeVar

T = TypeVar("T")
_DONE = object()


class _Failure:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class BackgroundLoop:
    """A persistent asyncio event loop running in a daemon thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="agent-loop", daemon=True)
        self._thread.start()

    def iterate(self, agen_factory: Callable[[], AsyncIterator[T]], *, timeout: float = 900) -> Iterator[T]:
        """Run an async generator on the loop and yield its items synchronously, as they arrive.

        Exceptions raised inside the generator are re-raised in the caller's thread.
        """
        q: queue.Queue[object] = queue.Queue()

        async def pump() -> None:
            try:
                async for item in agen_factory():
                    q.put(item)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
                q.put(_Failure(exc))
            finally:
                q.put(_DONE)

        future = asyncio.run_coroutine_threadsafe(pump(), self.loop)
        try:
            while True:
                item = q.get(timeout=timeout)
                if item is _DONE:
                    break
                if isinstance(item, _Failure):
                    raise item.exc
                yield item  # type: ignore[misc]
        finally:
            if not future.done():
                future.cancel()
