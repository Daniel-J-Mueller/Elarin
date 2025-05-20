import logging
from pathlib import Path
from typing import Optional


_file_handler: Optional[logging.Handler] = None


def enable_file_logging(log_dir: str) -> None:
    """Save logs to ``log_dir``/debug.log in addition to STDOUT."""
    global _file_handler
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "debug.log"
    handler = logging.FileHandler(log_path)
    fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    _file_handler = handler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    The logger prints timestamped messages to STDOUT. Repeated calls with the
    same ``name`` will return the same logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        if _file_handler:
            logger.addHandler(_file_handler)
        logger.setLevel(level)
    return logger
