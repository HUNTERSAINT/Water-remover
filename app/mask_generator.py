"""Build tight pixel masks for watermark text.

Instead of blanking whole OCR rectangles (which destroys the background), the
mask follows only the thin letter strokes inside each region, grown by a few
pixels so the soft anti-aliased edges are included too.
"""
from __future__ import annotations

import cv2
import numpy as np

from .watermark_detector import TextRegion, stroke_response


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _stroke_mask(gray_crop: np.ndarray, ksizes: list[int], keep: np.ndarray | None):
    resp = np.zeros_like(gray_crop)
    for k in ksizes:
        resp = np.maximum(resp, stroke_response(gray_crop, _odd(max(3, k))))
    ref = resp[keep] if keep is not None and keep.any() else resp.ravel()
    if ref.size == 0 or ref.max() < 3:
        return np.zeros_like(gray_crop, dtype=np.uint8)
    otsu, _ = cv2.threshold(ref.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(3.0, 0.55 * otsu)
    m = (resp > thr).astype(np.uint8) * 255
    if keep is not None:
        m[~keep] = 0
    return m


def region_mask(
    gray: np.ndarray,
    region: TextRegion | None = None,
    box: tuple[int, int, int, int] | None = None,
    strength: float = 1.0,
    fill_if_empty: bool = False,
) -> np.ndarray:
    """Return a full-size uint8 mask (255 = remove) for one region or box."""
    h_img, w_img = gray.shape[:2]
    out = np.zeros((h_img, w_img), np.uint8)
    if region is not None:
        box = region.box
    assert box is not None
    x0, y0, x1, y1 = box
    th = max(6, y1 - y0)
    pad = int(th * 0.30) + 3
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(w_img, x1 + pad), min(h_img, y1 + pad)
    crop = gray[Y0:Y1, X0:X1]

    keep = np.zeros(crop.shape, bool)
    if region is not None:
        poly = region.poly.astype(np.int32) - np.array([X0, Y0], np.int32)
        tmp = np.zeros(crop.shape, np.uint8)
        cv2.fillPoly(tmp, [poly], 255)
        grow = _odd(int(th * 0.15) + 1)
        tmp = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        keep = tmp > 0
        ksizes = [max(7, int(th * 0.24))]
    else:
        keep[:] = True
        ksizes = [9, 17, 31]

    m = _stroke_mask(crop, ksizes, keep)

    area = max(1, int(keep.sum()))
    if m.sum() / 255 < 0.015 * area:
        m[:] = 0
        if region is not None or fill_if_empty:
            if region is not None:
                cv2.fillPoly(m, [poly], 255)
            else:
                m[:] = 255

    d = 2 * max(1, int(round(th * 0.035 * strength))) + 1
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    out[Y0:Y1, X0:X1] = m
    return out


def build_mask(
    bgr: np.ndarray,
    regions: list[TextRegion],
    manual_boxes: list[tuple[int, int, int, int]] | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for r in regions:
        mask = np.maximum(mask, region_mask(gray, region=r, strength=strength))
    for b in manual_boxes or []:
        mask = np.maximum(mask, region_mask(gray, box=b, strength=strength, fill_if_empty=True))
    return mask
