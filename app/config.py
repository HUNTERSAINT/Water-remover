"""Runtime configuration, read from environment variables only."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    pass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class Settings:
    token: str
    max_image_mb: int
    max_megapixels: int
    max_concurrent_jobs: int
    job_timeout_s: int
    session_ttl_s: int
    temp_root: Path
    use_lama: bool
    lama_model_path: Path
    lama_model_url: str
    auto_crop: bool
    output_quality: int
    ocr_max_side: int
    health_port: int | None
    log_level: str

    @property
    def max_image_bytes(self) -> int:
        return self.max_image_mb * 1024 * 1024

    @property
    def max_pixels(self) -> int:
        return self.max_megapixels * 1_000_000


def load_settings(require_token: bool = True) -> Settings:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if require_token and not token:
        raise ConfigError(
            "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and "
            "put the token in your hosting's environment variables."
        )

    default_temp = Path(tempfile.gettempdir()) / "ai-image-cleaner"
    port_raw = os.getenv("PORT") or os.getenv("HEALTH_PORT")
    health_port = None
    if port_raw and port_raw.isdigit():
        health_port = int(port_raw)

    return Settings(
        token=token,
        max_image_mb=_int("MAX_IMAGE_MB", 15, 1, 19),  # Bot API download cap is 20 MB
        max_megapixels=_int("MAX_MEGAPIXELS", 40, 1, 200),
        max_concurrent_jobs=_int("MAX_CONCURRENT_JOBS", 2, 1, 8),
        job_timeout_s=_int("JOB_TIMEOUT_SECONDS", 120, 10, 900),
        session_ttl_s=_int("SESSION_TTL_SECONDS", 900, 60, 86400),
        temp_root=Path(os.getenv("TEMP_DIR") or default_temp),
        use_lama=_bool("USE_LAMA", False),
        lama_model_path=Path(os.getenv("LAMA_MODEL_PATH") or "models/lama_fp32.onnx"),
        lama_model_url=os.getenv(
            "LAMA_MODEL_URL",
            "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        ),
        auto_crop=_bool("AUTO_CROP_SCREENSHOT_BORDERS", True),
        output_quality=_int("OUTPUT_QUALITY", 95, 60, 100),
        ocr_max_side=_int("OCR_MAX_SIDE", 1600, 640, 4000),
        health_port=health_port,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
