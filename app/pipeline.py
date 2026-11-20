"""Orchestrates: validate -> detect -> mask -> inpaint -> verify -> encode."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import image_optimizer, image_receiver, mask_generator, watermark_detector
from .config import Settings
from .image_inpainting import Cancelled
from .image_receiver import LoadedImage
from .watermark_detector import Detection, TextRegion

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]


@dataclass
class Analysis:
    loaded: LoadedImage
    detection: Detection


@dataclass
class Options:
    strength: float = 1.0
    selected: list[TextRegion] | None = None   # explicit region choice
    manual_boxes: list[Box] = field(default_factory=list)
    include_uncertain: bool = False


@dataclass
class Outcome:
    data: bytes
    ext: str
    mime: str
    engine: str
    report: dict
    regions_removed: int
    traces_may_remain: bool


def analyse(data: bytes, cfg: Settings, cancel: threading.Event | None = None) -> Analysis:
    loaded = image_receiver.load_image(
        data, cfg.max_image_bytes, cfg.max_pixels, auto_crop=cfg.auto_crop
    )
    _check(cancel)
    detection = watermark_detector.detect(loaded.bgr, cfg.ocr_max_side)
    return Analysis(loaded, detection)


def default_targets(det: Detection, include_uncertain: bool = False) -> list[TextRegion]:
    return det.watermarks + (det.uncertain if include_uncertain else [])


def remove(
    analysis: Analysis,
    options: Options,
    inpainter,
    cfg: Settings,
    cancel: threading.Event | None = None,
) -> Outcome:
    loaded, det = analysis.loaded, analysis.detection
    targets = options.selected if options.selected is not None else default_targets(
        det, options.include_uncertain
    )
    if not targets and not options.manual_boxes:
        raise ValueError("nothing selected")

    _check(cancel)
    mask = mask_generator.build_mask(
        loaded.bgr, targets, options.manual_boxes, strength=options.strength
    )
    if not mask.any():
        raise ValueError("empty mask")

    _check(cancel)
    result = inpainter.inpaint(loaded.bgr, mask, cancel)

    # Guarantee: nothing outside the mask changes, even if an engine misbehaves.
    result[mask == 0] = loaded.bgr[mask == 0]

    report = image_optimizer.compare(loaded.bgr, result, mask)
    _check(cancel)
    data, ext, mime = image_optimizer.encode(result, loaded, cfg.output_quality)
    traces = report["residual_ratio"] > 0.5
    log.info("Removed %d regions with %s: %s", len(targets) + len(options.manual_boxes),
             inpainter.name, report)
    return Outcome(
        data, ext, mime, inpainter.name, report,
        len(targets) + len(options.manual_boxes), traces,
    )


def annotated_preview(analysis: Analysis, selected_ids: set[int] | None = None,
                      max_side: int = 1280) -> bytes:
    """JPEG preview with numbered boxes over detected text, for confirmation."""
    img = analysis.loaded.bgr.copy()
    h, w = img.shape[:2]
    s = max(1.0, max(h, w) / 900.0)
    thick = max(2, int(round(2 * s)))
    for i, r in enumerate(analysis.detection.regions, 1):
        if r.kind == watermark_detector.PROTECTED:
            continue
        on = selected_ids is None or i in selected_ids
        color = (60, 200, 60) if (r.kind == watermark_detector.WATERMARK and on) else (0, 165, 255)
        if not on:
            color = (160, 160, 160)
        x0, y0, x1, y1 = r.box
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thick)
        label = str(i)
        fs = 0.9 * s
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        ty = max(th + 4, y0 - 4)
        cv2.rectangle(img, (x0, ty - th - 6), (x0 + tw + 10, ty + 4), color, -1)
        cv2.putText(img, label, (x0 + 5, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), thick, cv2.LINE_AA)
    sc = min(1.0, max_side / float(max(h, w)))
    if sc < 1.0:
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def grid_boxes(size: tuple[int, int], cols: int = 4, rows: int = 4) -> list[Box]:
    w, h = size
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append((c * w // cols, r * h // rows, (c + 1) * w // cols, (r + 1) * h // rows))
    return out


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def grid_preview(analysis: Analysis, selected: set[int], cols: int = 4, rows: int = 4,
                 max_side: int = 1280) -> bytes:
    """Image with a numbered grid; selected cells are tinted green."""
    img = analysis.loaded.bgr.copy()
    h, w = img.shape[:2]
    s = max(1.0, max(h, w) / 900.0)
    overlay = img.copy()
    for i, (x0, y0, x1, y1) in enumerate(grid_boxes((w, h), cols, rows), 1):
        if i in selected:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (60, 200, 60), -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    for i, (x0, y0, x1, y1) in enumerate(grid_boxes((w, h), cols, rows), 1):
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), max(1, int(s)))
        cv2.putText(img, str(i), (x0 + 8, y0 + int(34 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 * s, (0, 0, 0), max(3, int(4 * s)), cv2.LINE_AA)
        cv2.putText(img, str(i), (x0 + 8, y0 + int(34 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    sc = min(1.0, max_side / float(max(h, w)))
    if sc < 1.0:
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""
