"""
common/run_logging.py

Logging + run directory helpers.

- get_logger(): file + console logger; console prints nicely with tqdm progress bars.
- get_save_dir(): create a unique run directory under base_dir/train|test/name-XX

Logic intentionally kept consistent with your original implementation.
"""

from __future__ import annotations

import logging
import os

import tqdm


class StreamHandlerWithTQDM(logging.Handler):
    """Logging handler compatible with tqdm progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def get_logger(log_dir: str, name: str) -> logging.Logger:
    """Create a logger that logs to both console and a file.

    Args:
        log_dir: Directory in which to create the log file.
        name: Logger name.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler (DEBUG+)
    log_path = os.path.join(log_dir, "log.txt")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Console handler (INFO+), tqdm-friendly
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m.%d.%y %H:%M:%S")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_save_dir(base_dir: str, name: str, training: bool, id_max: int = 100) -> str:
    """Get a unique save directory under base_dir/(train|test)/name-XX."""
    for uid in range(1, int(id_max)):
        subdir = "train" if training else "test"
        save_dir = os.path.join(base_dir, subdir, f"{name}-{uid:02d}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError(
        "Too many save directories created with the same name. "
        "Delete old save directories or use another name."
    )
