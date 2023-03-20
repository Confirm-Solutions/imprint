import logging
import sys


def configure_logging(is_testing=False):
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("confirm").setLevel(logging.DEBUG)
    logging.getLogger("imprint").setLevel(logging.DEBUG)
    if is_testing:
        logging.getLogger("tests").setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.NOTSET)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s \n" "%(message)s"
    )
    handler.setFormatter(formatter)
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(handler)
