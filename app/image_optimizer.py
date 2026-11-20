"""Final encoding and result verification."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from .image_receiver import LoadedImage
from .watermark_detector import stroke_response

MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def encode(bgr: np.ndarray, loaded: LoadedImage, quality: int = 95) -> tuple[bytes, str, str]:
    """Encode in the same format as the upload, at high quality."""
    rgb = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
    fmt = loaded.fmt
    if loaded.alpha is not None and fmt in ("PNG", "WEBP"):
        rgb = rgb.convert("RGBA")
        rgb.putalpha(Image.fromarray(loaded.alpha))
    buf = io.BytesIO()
    kwargs: dict = {}
    if loaded.icc_profile:
        kwargs["icc_profile"] = loaded.icc_profile
    if fmt == "JPEG":
        rgb.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=0, optimize=True, **kwargs)
    elif fmt == "PNG":
        rgb.save(buf, "PNG", optimize=True, compress_level=6, **kwargs)
    else:
        rgb.save(buf, "WEBP", quality=quality, method=4, **kwargs)
    return buf.getvalue(), EXT[fmt], MIME[fmt]


def compare(original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> dict:
    """Check the edit stayed inside the mask and measure leftover marks."""
    keep = mask == 0
    outside_diff = int(np.abs(original.astype(np.int16) - result.astype(np.int16))[keep].max()) if keep.any() else 0
    inside = mask > 0
    changed = float(np.abs(original.astype(np.int16) - result.astype(np.int16))[inside].mean()) if inside.any() else 0.0

    g0 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Look just inside the mask footprint for leftover thin structures.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foot = cv2.dilate(mask, k) > 0
    r0 = stroke_response(g0, 9)[foot].mean() if foot.any() else 0.0
    r1 = stroke_response(g1, 9)[foot].mean() if foot.any() else 0.0
    residual = float(r1 / r0) if r0 > 1e-6 else 0.0
    return {
        "outside_max_diff": outside_diff,
        "mean_change_inside": round(changed, 2),
        "mask_fraction": round(float(inside.mean()), 5),
        "residual_ratio": round(residual, 3),
    }
