"""Logging helpers for sweets.

Logging is handled by `loguru`. Modules should import the logger directly::

    from loguru import logger

    logger.info("Something happened")
    logger.success("Something great happened: highlight this success")

This module only provides the `log_runtime` decorator for timing functions;
loguru's default stderr handler is used as-is.
"""

import time
from collections.abc import Callable
from functools import wraps

from loguru import logger

__all__ = ["log_runtime"]


def log_runtime(f: Callable) -> Callable:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = f(*args, **kwargs)
        elapsed_seconds = time.time() - t1
        elapsed_minutes = elapsed_seconds / 60.0
        logger.info(
            f"Total elapsed time for {f.__module__}.{f.__name__} : "
            f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.2f} seconds)"
        )
        return result

    return wrapper
