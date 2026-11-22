"""Process setup required before the async database pool is used."""

from __future__ import annotations

import asyncio
import selectors
import sys
import warnings
from collections.abc import Callable


def _selector_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def install_compatible_event_loop() -> None:
    """Use a selector loop on Windows so async psycopg can open connections.

    Uvicorn's default Windows loop is a Proactor loop, which psycopg refuses.
    Importing the app happens before Uvicorn chooses its loop, so replacing
    that factory here keeps ``uvicorn resume_reader.main:app`` working.
    """
    if sys.platform != "win32":
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        policy_type = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy_type is not None:
            asyncio.set_event_loop_policy(policy_type())

    try:
        import uvicorn.loops.asyncio as uvicorn_loops
    except ImportError:
        return

    def asyncio_loop_factory(
        use_subprocess: bool = False,
    ) -> Callable[[], asyncio.AbstractEventLoop]:
        # Uvicorn passes this flag; the selector loop is required either way.
        del use_subprocess
        return _selector_loop

    uvicorn_loops.asyncio_loop_factory = asyncio_loop_factory
