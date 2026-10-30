"""Execution-time measurement (TP3 ex.3 point 4).

Correction w.r.t. TP3: ``time.time()`` is a wall-clock that can jump and has a coarse
resolution on some systems; ``time.perf_counter()`` is the right clock for durations.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def timed(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call ``func(*args, **kwargs)`` and return ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start
