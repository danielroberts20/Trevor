"""
Logging configuration for Trevor.

Matches TravelNet's logging conventions:
  - Format: %(asctime)s | %(levelname)s | %(name)s | %(message)s
  - Coloured stdout output
  - paramiko.transport silenced to WARNING (SSH auth/connect spam)
  - uvicorn.access silenced to WARNING
"""

import logging
import sys


class ColouredFormatter(logging.Formatter):
    COLOURS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        colour = self.COLOURS.get(record.levelname, "")
        # Copy the record to avoid mutating the original (shared across handlers)
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"{colour}{record.levelname}{self.RESET}"
        return super().format(record)


def configure_logging():
    """Configure root logger with coloured stdout output.

    Silences noisy third-party loggers that produce INFO spam:
      - paramiko.transport: SSH connection/auth messages on every poll cycle
      - uvicorn.access: per-request access log lines
    """
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ColouredFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
