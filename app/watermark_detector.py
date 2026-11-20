"""Find text overlays and decide which of them look like watermarks.

Detection is OCR based (RapidOCR / PaddleOCR models running on onnxruntime,
CPU only). Every text box is then scored on how "watermark-like" it is:

* low stroke contrast against the background (semi-transparent overlay)
  versus crisp, high-contrast print (product labels, serial numbers);
* known watermark wording (site names, "posted on", copyright marks...);
* the same text repeated across the image (tiled watermarks).

Opaque, high-contrast text is classed as *protected* and is never removed
automatically.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

WATERMARK_WORDS = re.compile(
    r"(jiji|posted\s*on|shutterstock|getty|alamy|istock|dreamstime|adobe\s*stock|"
    r"depositphotos|123rf|watermark|copyright|©|\(c\)|preview|sample|www\.|\.com|"
    r"\.ng\b|\.net|\.org|https?:|^@\w+|instagram|facebook|tiktok|snapchat|"
    r"all\s*rights|reserved|stock\s*photo)",
    re.IGNORECASE,
)

WATERMARK, UNCERTAIN, PROTECTED = "watermark", "uncertain", "protected"


@dataclass
class TextRegion:
    box: tuple[int, int, int, int]       # x0, y0, x1, y1 (image coordinates)
    poly: np.ndarray                     # 4x2 int32 polygon
    text: str
    ocr_conf: float
    contrast: float = 0.0                # median stroke contrast, 0..255
    score: float = 0.0                   # watermark likelihood 0..1
    kind: str = UNCERTAIN
    reasons: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass
class Detection:
    regions: list[TextRegion]
    image_size: tuple[int, int]          # (w, h)

    def of_kind(self, kind: str) -> list[TextRegion]:
        return [r for r in self.regions if r.kind == kind]

    @property
    def watermarks(self) -> list[TextRegion]:
        return self.of_kind(WATERMARK)

    @property
    def uncertain(self) -> list[TextRegion]:
        return self.of_kind(UNCERTAIN)

    @property
    def protected(self) -> list[TextRegion]:
        return self.of_kind(PROTECTED)


# --------------------------------------------------------------------------
# OCR engine (lazy singleton, guarded by a lock: onnxruntime sessions are
# fine across threads but RapidOCR keeps small per-call state)
# --------------------------------------------------------------------------
_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            log.info("Loading OCR models")
            _engine = RapidOCR()
        return _engine


def _run_ocr(img_bgr: np.ndarray, text_score: float = 0.35):
    engine = _get_engine()
    with _engine_lock:
        result, _ = engine(img_bgr, text_score=text_score)
    return result or []


def _enhance_faint(gray: np.ndarray) -> np.ndarray:
    """Boost faint overlays so OCR can see them: subtract a heavy blur and
    stretch what is left (this removes smooth background, keeps thin marks)."""
    g = gray.astype(np.float32)
    k = max(15, (min(gray.shape) // 25) | 1)
    detail = g - cv2.GaussianBlur(g, (k, k), 0)
    detail = np.clip(detail * 6.0 + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / float(ua)


def _items_to_regions(items, scale: float, shape) -> list[TextRegion]:
    h, w = shape[:2]
    out: list[TextRegion] = []
    for poly, text, conf in items:
        pts = (np.array(poly, dtype=np.float32) / scale).round().astype(np.int32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        out.append(
            TextRegion((x0, y0, x1, y1), pts, str(text).strip(), float(conf))
        )
    return out


def stroke_response(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Magnitude of thin bright-or-dark structures narrower than ``ksize``."""
    ksize = max(3, ksize | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    return np.maximum(dark, bright)


def measure_contrast(gray: np.ndarray, box) -> float:
    """Median stroke contrast inside a text box (0 = invisible, 255 = black on
    white). Opaque printed text scores high, translucent overlays low."""
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    k = max(5, int(min(crop.shape[:2]) * 0.45))
    resp = stroke_response(crop, k)
    if resp.max() < 3:
        return 0.0
    thr, _ = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strokes = resp > max(thr, 3)
    if strokes.sum() < 10:
        return float(resp.max())
    return float(np.percentile(resp[strokes], 75))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def classify(regions: list[TextRegion], gray: np.ndarray) -> None:
    # Count a text as "repeated" only when the copies sit well apart; the
    # same words detected twice in one place are just OCR duplicates.
    groups: dict[str, list[TextRegion]] = {}
    for r in regions:
        n = _norm(r.text)
        if n:
            groups.setdefault(n, []).append(r)

    def repeat_count(r: TextRegion) -> int:
        n = _norm(r.text)
        cx, cy = (r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2
        far = 0
        for o in groups.get(n, []):
            if o is r:
                continue
            ox, oy = (o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2
            if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 > 2.5 * max(r.height, o.height):
                far += 1
        return far + 1

    img_h, img_w = gray.shape[:2]

    for r in regions:
        r.contrast = measure_contrast(gray, r.box)
        score = 0.0
        if r.contrast <= 32:
            score += 0.62
            r.reasons.append("faint, semi-transparent strokes")
        elif r.contrast <= 55:
            score += 0.42
            r.reasons.append("moderately faint strokes")
        elif r.contrast <= 85:
            score += 0.18
        else:
            r.reasons.append("crisp high-contrast print")

        if WATERMARK_WORDS.search(r.text):
            score += 0.28
            r.reasons.append("watermark-style wording")
        n = _norm(r.text)
        if n and repeat_count(r) >= 2:
            score += 0.22
            r.reasons.append("repeated across the image")
        big = r.height >= 0.03 * img_h or (r.box[2] - r.box[0]) >= 0.35 * img_w
        if big and r.contrast <= 85:
            score += 0.12
            r.reasons.append("large overlay-sized text")
        if r.ocr_conf < 0.45:
            score -= 0.1

        r.score = float(max(0.0, min(1.0, score)))
        if r.score >= 0.5:
            r.kind = WATERMARK
        elif r.score >= 0.3:
            r.kind = UNCERTAIN
        else:
            r.kind = PROTECTED


def detect(bgr: np.ndarray, ocr_max_side: int = 1600) -> Detection:
    h, w = bgr.shape[:2]
    scale = min(1.0, ocr_max_side / float(max(h, w)))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else bgr

    regions = _items_to_regions(_run_ocr(small), scale, bgr.shape)

    # Second pass on a contrast-boosted copy to catch faint overlays that
    # plain OCR misses. Merge, preferring the first-pass boxes.
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    extra = _items_to_regions(_run_ocr(_enhance_faint(gray_small), 0.45), scale, bgr.shape)
    for cand in extra:
        if all(_iou(cand.box, r.box) < 0.4 for r in regions):
            regions.append(cand)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    classify(regions, gray)
    regions.sort(key=lambda r: (r.box[1], r.box[0]))
    log.info(
        "OCR found %d text regions: %d watermark, %d uncertain, %d protected",
        len(regions),
        len([r for r in regions if r.kind == WATERMARK]),
        len([r for r in regions if r.kind == UNCERTAIN]),
        len([r for r in regions if r.kind == PROTECTED]),
    )
    return Detection(regions, (w, h))
