import logging

_logger = logging.getLogger("taskb.say")


def say(msg: str) -> None:
    """Print a message and log it."""
    print(msg)
    _logger.info(msg)
