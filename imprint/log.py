import contextvars
import logging
import sys

# ContextVar stores context-local state. It is similar to thread-local state,
# but works for both asyncio coroutines and threads.
worker_id = contextvars.ContextVar("worker_id", default=None)


def configure_logging(is_testing=False):
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.worker_id = worker_id.get()
        return record

    logging.setLogRecordFactory(record_factory)

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("confirm").setLevel(logging.DEBUG)
    logging.getLogger("imprint").setLevel(logging.DEBUG)
    if is_testing:
        logging.getLogger("tests").setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.NOTSET)
    formatter = logging.Formatter(
        "[worker_id=%(worker_id)s] %(asctime)s - %(name)s - %(levelname)s \n"
        "%(message)s"
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
