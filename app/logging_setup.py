"""Logging that never leaks the bot token."""
from __future__ import annotations

import logging
import re

_TOKEN_RE = re.compile(r"(?<![0-9])\d{6,}:[A-Za-z0-9_-]{30,}")


class RedactTokenFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - never break logging
            return True
        if _TOKEN_RE.search(msg):
            record.msg = _TOKEN_RE.sub("<redacted-token>", msg)
            record.args = ()
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handler.addFilter(RedactTokenFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    # httpx logs full request URLs, which contain the bot token.
    for noisy in ("httpx", "httpcore", "telegram.ext.Application"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
