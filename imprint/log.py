import contextvars
import logging

import pandas as pd

# ContextVar stores context-local state. It is similar to thread-local state,
# but works for both asyncio coroutines and threads.
worker_id = contextvars.ContextVar("worker_id", default=None)


class Adapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        with pd.option_context("display.max_columns", None):
            return f"[worker_id={worker_id.get()}] \n{msg}", kwargs


def getLogger(name):
    """
    A replacement for logging.getLogger that adds the worker_id to the log message.

    Args:
        name: the name of the logger

    Returns:
        A LoggerAdapter that adds the worker_id to the log message.
    """
    return Adapter(logging.getLogger(name), {})
