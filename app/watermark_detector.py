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
    support: int = 1                     # how many OCR passes / fragments back this region
    enhanced_only: bool = False          # seen only on contrast-boosted passes
    neighbor: bool = False               # found next to a confirmed watermark line

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

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
# OCR engine (lazy singleton, guarded by a lock)
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


def _run_ocr(img_bgr: np.ndarray, text_score: float = 0.35,
             box_thresh: float = 0.5, unclip_ratio: float = 1.6):
    engine = _get_engine()
    with _engine_lock:
        # RapidOCR keeps these settings between calls, so always pass all three.
        result, _ = engine(img_bgr, text_score=text_score, box_thresh=box_thresh,
                           unclip_ratio=unclip_ratio)
    return result or []


def _enhance_faint(gray: np.ndarray) -> np.ndarray:
    """Subtract a heavy blur and stretch what is left: removes smooth background,
    keeps thin marks."""
    g = gray.astype(np.float32)
    k = max(15, (min(gray.shape) // 25) | 1)
    detail = g - cv2.GaussianBlur(g, (k, k), 0)
    detail = np.clip(detail * 6.0 + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)


def _dark_enhance(bgr: np.ndarray, sigma: float, gain: float = 6.0) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = g - cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.cvtColor(np.clip(128 + d * gain, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / float(ua)


def _alnum(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _items_to_regions(items, scale: float, shape, enhanced: bool = False,
                      offset: tuple[int, int] = (0, 0), pad: int = 0) -> list[TextRegion]:
    h, w = shape[:2]
    ox, oy = offset
    out: list[TextRegion] = []
    for poly, text, conf in items:
        text = str(text).strip()
        conf = float(conf)
        if enhanced and (conf < 0.5 or _alnum(text) < 3):
            continue                       # contrast-boosted passes hallucinate on texture
        pts = (np.array(poly, dtype=np.float32) - pad) / scale
        pts = (pts + np.array([ox, oy], np.float32)).round().astype(np.int32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        pts[:, 0] = pts[:, 0].clip(0, w - 1)
        pts[:, 1] = pts[:, 1].clip(0, h - 1)
        out.append(TextRegion((x0, y0, x1, y1), pts, text, conf,
                              enhanced_only=enhanced))
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


# ----------------------------------------------------------- collection
def _ocr_variants(small: np.ndarray):
    """Several views of the same image. Faint overlays are often only readable
    on an upscaled, contrast-boosted copy, so small images get extra passes."""
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    long_side = max(small.shape[:2])
    variants = [
        (small, 1.0, dict(text_score=0.35), False),
        (_enhance_faint(gray), 1.0, dict(text_score=0.45), True),
    ]
    if long_side <= 1100:
        for sc in (2, 3):
            big = cv2.resize(small, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            variants.append((_dark_enhance(big, 4.0 * sc), float(sc),
                             dict(text_score=0.3, box_thresh=0.3, unclip_ratio=1.8), True))
    return variants


def _axis_aligned(poly: np.ndarray, tol_deg: float = 8.0) -> bool:
    (x0, y0), (x1, y1) = poly[0], poly[1]
    ang = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    return ang < tol_deg or abs(ang - 180) < tol_deg


def _same_line(a: TextRegion, b: TextRegion) -> bool:
    inter = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
    if inter <= 0 or inter / float(min(a.height, b.height)) < 0.6:
        return False
    gap = max(a.box[0], b.box[0]) - min(a.box[2], b.box[2])
    if gap > 1.2 * max(a.height, b.height):
        return False
    if max(a.height, b.height) > 1.8 * min(a.height, b.height):
        return False
    if not (_axis_aligned(a.poly) and _axis_aligned(b.poly)):
        return False
    return (a.contrast <= 85) == (b.contrast <= 85)


def _union(a: TextRegion, b: TextRegion) -> TextRegion:
    x0, y0 = min(a.box[0], b.box[0]), min(a.box[1], b.box[1])
    x1, y1 = max(a.box[2], b.box[2]), max(a.box[3], b.box[3])
    first, second = sorted((a, b), key=lambda r: r.box[0])
    text = first.text if second.text in first.text else (
        second.text if first.text in second.text else f"{first.text} {second.text}")
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
    return TextRegion((x0, y0, x1, y1), poly, text, max(a.ocr_conf, b.ocr_conf),
                      contrast=min(a.contrast, b.contrast), support=a.support + b.support,
                      enhanced_only=a.enhanced_only and b.enhanced_only,
                      neighbor=a.neighbor or b.neighbor)


def _overlap_frac(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(smaller) if smaller else 0.0


def _dedupe(regions: list[TextRegion]) -> list[TextRegion]:
    """The same text seen by several OCR passes: keep one box (preferring the
    plain pass, then the most confident) rather than growing a union."""
    def pref(r):
        return (not r.enhanced_only, r.ocr_conf)

    kept: list[TextRegion] = []
    for r in sorted(regions, key=pref, reverse=True):
        for k in kept:
            if _overlap_frac(r.box, k.box) > 0.6 and 0.5 < r.height / float(k.height) < 2.0:
                k.support += r.support
                break
        else:
            kept.append(r)
    return kept


def _merge_lines(regions: list[TextRegion]) -> list[TextRegion]:
    regions = _dedupe(regions)
    changed = True
    while changed:
        changed = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if _same_line(regions[i], regions[j]):
                    regions[i] = _union(regions[i], regions[j])
                    del regions[j]
                    changed = True
                    break
            if changed:
                break
    return regions


def _collect(img: np.ndarray, gray: np.ndarray, ocr_max_side: int,
             offset: tuple[int, int] = (0, 0)) -> list[TextRegion]:
    """Run all OCR variants over ``img`` and return merged regions in the
    coordinates of ``img`` (plus ``offset``)."""
    h, w = img.shape[:2]
    scale = min(1.0, ocr_max_side / float(max(h, w)))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else img
    found: list[TextRegion] = []
    for view, sc, kw, enhanced in _ocr_variants(small):
        found += _items_to_regions(_run_ocr(view, **kw), scale * sc, img.shape,
                                   enhanced=enhanced, offset=(0, 0))
    full_h, full_w = gray.shape[:2]
    for r in found:
        r.contrast = measure_contrast(gray, r.box)
    return _merge_lines(found)


# ------------------------------------------------------------- scoring
def classify(regions: list[TextRegion], gray: np.ndarray) -> None:
    # Count a text as "repeated" only when copies sit well apart; the same
    # words seen twice in one place are just OCR duplicates.
    groups: dict[str, list[TextRegion]] = {}
    for r in regions:
        n = _norm(r.text)
        if n:
            groups.setdefault(n, []).append(r)

    def repeat_count(r: TextRegion) -> int:
        cx, cy = (r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2
        far = 0
        for o in groups.get(_norm(r.text), []):
            if o is r:
                continue
            ox, oy = (o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2
            if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 > 2.5 * max(r.height, o.height):
                far += 1
        return far + 1

    img_h, img_w = gray.shape[:2]
    for r in regions:
        if r.contrast == 0.0:
            r.contrast = measure_contrast(gray, r.box)
        r.reasons = []
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

        wording = bool(WATERMARK_WORDS.search(r.text))
        repeated = bool(_norm(r.text)) and repeat_count(r) >= 2
        big = (r.height >= 0.06 * img_h or r.width >= 0.30 * img_w) and r.contrast <= 85
        if wording:
            score += 0.28
            r.reasons.append("watermark-style wording")
        if repeated:
            score += 0.22
            r.reasons.append("repeated across the image")
        if big:
            score += 0.12
            r.reasons.append("large overlay-sized text")
        if r.neighbor and r.contrast <= 85:
            score += 0.3
            r.reasons.append("stacked next to a watermark line")
        if r.ocr_conf < 0.45:
            score -= 0.1

        # Faint strokes alone are not enough to delete something automatically.
        corroborated = wording or repeated or big or r.neighbor
        if not corroborated:
            score = min(score, 0.45)

        r.score = float(max(0.0, min(1.0, score)))
        if r.score >= 0.5:
            r.kind = WATERMARK
        elif r.score >= 0.3:
            r.kind = UNCERTAIN
        else:
            r.kind = PROTECTED


# ---------------------------------------------------- neighbour search
def _search_neighbors(bgr: np.ndarray, gray: np.ndarray, regions: list[TextRegion]) -> list[TextRegion]:
    """Watermarks often come as stacked lines (e.g. "POSTED ON ..." above a shop
    name). Search a band above and below each confirmed line with a more
    sensitive, upscaled OCR pass."""
    H, W = gray.shape[:2]
    added: list[TextRegion] = []
    for seed in [r for r in regions if r.kind == WATERMARK]:
        x0, y0, x1, y1 = seed.box
        h, w = seed.height, seed.width
        for above in (True, False):
            if above:
                by0, by1 = int(y0 - 2.2 * h), int(y0 - 0.1 * h)
            else:
                by0, by1 = int(y1 + 0.1 * h), int(y1 + 2.2 * h)
            bx0, bx1 = int(x0 - 0.15 * w), int(x1 + 0.15 * w)
            by0, by1, bx0, bx1 = max(0, by0), min(H, by1), max(0, bx0), min(W, bx1)
            if by1 - by0 < 12 or bx1 - bx0 < 40:
                continue
            crop = bgr[by0:by1, bx0:bx1]
            sc = float(np.clip(160.0 / crop.shape[0], 2.0, 6.0))
            big = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            pad = 40
            big = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            items = _run_ocr(big, text_score=0.25, box_thresh=0.25, unclip_ratio=1.8)
            cands = _items_to_regions(items, sc, bgr.shape, offset=(bx0, by0), pad=pad)
            for c in cands:
                if _alnum(c.text) < 3 or c.ocr_conf < 0.45:
                    continue
                if not (by0 <= (c.box[1] + c.box[3]) / 2 <= by1):
                    continue
                if any(_iou(c.box, r.box) > 0.25 for r in regions + added):
                    continue
                c.contrast = measure_contrast(gray, c.box)
                if c.contrast > 85:
                    continue
                c.neighbor = True
                added.append(c)
    return added


def detect(bgr: np.ndarray, ocr_max_side: int = 1600) -> Detection:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    regions = _collect(bgr, gray, ocr_max_side)
    classify(regions, gray)
    extra = _search_neighbors(bgr, gray, regions)
    if extra:
        regions = _merge_lines(regions + extra)
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


def detect_in_box(bgr: np.ndarray, box: tuple[int, int, int, int],
                  ocr_max_side: int = 1600) -> Detection:
    """Sensitive search inside an area the user marked by hand."""
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = box
    pad = 10
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = np.ascontiguousarray(bgr[Y0:Y1, X0:X1])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    padv = 0
    if crop.shape[0] < 64:                       # OCR skips very short images
        padv = (64 - crop.shape[0]) // 2 + 1
        crop = cv2.copyMakeBorder(crop, padv, padv, 0, 0, cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    regions = _collect(crop, gray, ocr_max_side)
    out = []
    for r in regions:
        bx0, by0, bx1, by1 = r.box
        r.box = (bx0 + X0, max(0, by0 - padv) + Y0, bx1 + X0, max(0, by1 - padv) + Y0)
        r.poly = (r.poly - np.array([0, padv], np.int32) + np.array([X0, Y0], np.int32)).astype(np.int32)
        out.append(r)
    return Detection(out, (W, H))
