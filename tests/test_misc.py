import logging

import pytest

from app.config import ConfigError, load_settings
from app.logging_setup import RedactTokenFilter


def test_token_required(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(ConfigError):
        load_settings()


def test_env_values_are_clamped(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("MAX_IMAGE_MB", "999")
    assert load_settings().max_image_mb == 19


def test_token_is_redacted_from_logs():
    rec = logging.LogRecord("x", logging.INFO, "", 0,
                            "GET https://api.telegram.org/bot123456789:AAEhBP0av28v_aaaaaaaaaaaaaaaaaaaaaaaa/getMe", (), None)
    RedactTokenFilter().filter(rec)
    assert "AAEhBP0av28v" not in rec.getMessage()
