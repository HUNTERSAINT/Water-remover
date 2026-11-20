"""Validate and decode uploaded images.

Everything here works on bytes in memory; nothing is written to disk and the
caller's original bytes are never modified.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger(__name__)

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageError(Exception):
    """An error whose message is safe to show to the user."""


@dataclass
class LoadedImage:
    bgr: np.ndarray                      # working image (possibly border-cropped)
    fmt: str                             # JPEG | PNG | WEBP
    alpha: np.ndarray | None             # uint8 alpha for the working image, if any
    icc_profile: bytes | None
    crop_box: tuple[int, int, int, int] | None  # (x0, y0, x1, y1) in the full frame
    full_size: tuple[int, int]           # (w, h) of the decoded, uncropped image

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.bgr.shape[:2]
        return w, h


def load_image(
    data: bytes | bytearray,
    max_bytes: int,
    max_pixels: int,
    auto_crop: bool = True,
) -> LoadedImage:
    if not data:
        raise ImageError("The file is empty.")
    if len(data) > max_bytes:
        raise ImageError(
            f"That file is too large (limit {max_bytes // (1024 * 1024)} MB). "
            "Please send a smaller image."
        )

    try:
        probe = Image.open(io.BytesIO(data))
        fmt = (probe.format or "").upper()
        width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageError("I couldn't read that file. Is it a valid image?") from exc

    if fmt not in ALLOWED_FORMATS:
        raise ImageError("Unsupported format. Please send a JPG, PNG or WEBP image.")
    if width * height > max_pixels:
        raise ImageError(
            f"That image is too large ({width}x{height}). "
            f"The limit is {max_pixels // 1_000_000} megapixels."
        )
    if width < 32 or height < 32:
        raise ImageError("That image is too small to work with.")

    try:
        img = Image.open(io.BytesIO(data))
        icc = img.info.get("icc_profile")
        img.load()
        img = ImageOps.exif_transpose(img)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageError("That image looks corrupted and couldn't be decoded.") from exc

    alpha = None
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A")).copy()
        if alpha.min() == 255:
            alpha = None
        img = rgba.convert("RGB")
    else:
        img = img.convert("RGB")

    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    full_size = (bgr.shape[1], bgr.shape[0])

    crop_box = None
    if auto_crop:
        crop_box = find_content_box(bgr)
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            bgr = np.ascontiguousarray(bgr[y0:y1, x0:x1])
            if alpha is not None:
                alpha = np.ascontiguousarray(alpha[y0:y1, x0:x1])
            log.info("Cropped screenshot borders to %s", crop_box)

    return LoadedImage(bgr, fmt, alpha, icc, crop_box, full_size)


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    best, start = None, None
    for i, f in enumerate(np.append(flags, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def find_content_box(
    bgr: np.ndarray, black: int = 20, min_removed: float = 0.08
) -> tuple[int, int, int, int] | None:
    """Detect a photo sitting inside flat black screenshot chrome.

    Only crops when the photo is bordered by *perfectly flat* near-black
    strips, which is what phone screenshots of galleries/listings have and
    what dark natural photos (which have sensor noise) do not.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lit = gray > black

    rows = _longest_run(lit.mean(axis=1) > 0.6)
    if rows is None:
        return None
    y0, y1 = rows
    # Columns: anything that is not perfectly flat near-black is content.
    band = gray[y0:y1]
    content_cols = np.where((band.max(axis=0) > black) | (band.std(axis=0) > 3.0))[0]
    if content_cols.size == 0:
        return None
    x0, x1 = int(content_cols[0]), int(content_cols[-1]) + 1

    if (y1 - y0) < 0.2 * h or (x1 - x0) < 0.3 * w:
        return None
    removed = 1.0 - ((y1 - y0) * (x1 - x0)) / float(h * w)
    if removed < min_removed:
        return None

    def flat(strip: np.ndarray) -> bool:
        return strip.size > 0 and strip.max() <= black and strip.std() <= 3.0

    margin = 6
    checks = []
    if y0 > margin:
        checks.append(flat(gray[y0 - margin:y0, x0:x1]))
    if h - y1 > margin:
        checks.append(flat(gray[y1:y1 + margin, x0:x1]))
    if x0 > margin:
        checks.append(flat(gray[y0:y1, x0 - margin:x0]))
    if w - x1 > margin:
        checks.append(flat(gray[y0:y1, x1:x1 + margin]))
    if not checks or not all(checks):
        return None

    # Trim a couple of pixels so the black edge never leaks into the photo.
    x0, y0, x1, y1 = x0 + 1, y0 + 1, x1 - 1, y1 - 1
    return (x0, y0, x1, y1)
