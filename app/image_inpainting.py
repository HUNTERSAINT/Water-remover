"""Inpainting back-ends.

* ``LamaInpainter``  - optional LaMa ONNX model (best quality, CPU capable).
* ``OpenCVInpainter`` - always available, light, used as the fallback.

Both only ever change pixels inside the mask: the final composite copies every
unmasked pixel straight from the original image.
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


def _clusters(mask: np.ndarray, gap: int) -> list[tuple[int, int, int, int]]:
    """Group nearby mask pixels so each cluster is inpainted with local context."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    merged = cv2.dilate(mask, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


class OpenCVInpainter:
    name = "opencv"

    def inpaint(self, bgr, mask, cancel: threading.Event | None = None) -> np.ndarray:
        h, w = mask.shape
        result = bgr.copy()
        for (x0, y0, x1, y1) in _clusters(mask, gap=12):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            ctx = max(24, int(0.6 * max(x1 - x0, y1 - y0)))
            X0, Y0 = max(0, x0 - ctx), max(0, y0 - ctx)
            X1, Y1 = min(w, x1 + ctx), min(h, y1 + ctx)
            crop = bgr[Y0:Y1, X0:X1]
            m = mask[Y0:Y1, X0:X1]
            ch = int(max(y1 - y0, 6) // 8)
            radius = max(3, min(9, ch))
            # Telea keeps edges; blend with Navier-Stokes for smoother fills.
            a = cv2.inpaint(crop, m, radius, cv2.INPAINT_TELEA)
            b = cv2.inpaint(crop, m, radius, cv2.INPAINT_NS)
            filled = cv2.addWeighted(a, 0.5, b, 0.5, 0)
            filled = _reinject_grain(filled, crop, m)
            sel = m > 0
            region = result[Y0:Y1, X0:X1]
            region[sel] = filled[sel]
        return result


def _reinject_grain(filled: np.ndarray, orig: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Filled areas come out too smooth; add noise matching the surroundings so
    the patch does not show up as a blur."""
    ring = cv2.dilate(mask, np.ones((15, 15), np.uint8)) & ~cv2.dilate(
        mask, np.ones((5, 5), np.uint8)
    )
    sel = ring > 0
    if sel.sum() < 50:
        return filled
    g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.6)
    sigma = float(np.std(hp[sel]))
    sigma = min(sigma, 6.0)
    if sigma < 0.6:
        return filled
    rng = np.random.default_rng(1234)
    noise = rng.normal(0.0, sigma, size=filled.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
    out = filled.astype(np.float32) + noise[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


class LamaInpainter:
    """LaMa via onnxruntime. The Carve/LaMa-ONNX export is fixed at 512x512,
    so each mask cluster is processed in a 512 window (scaled when larger)."""

    name = "lama"
    SIZE = 512

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        ins = self.session.get_inputs()
        self.img_name, self.mask_name = ins[0].name, ins[1].name
        self.out_name = self.session.get_outputs()[0].name
        self._lock = threading.Lock()

    def _run(self, crop_bgr: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        s = self.SIZE
        h, w = crop_mask.shape
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb_s = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA if max(h, w) > s else cv2.INTER_CUBIC)
        m_s = cv2.resize(crop_mask, (s, s), interpolation=cv2.INTER_NEAREST)
        x = (rgb_s.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        m = (m_s > 0).astype(np.float32)[None, None]
        with self._lock:
            out = self.session.run([self.out_name], {self.img_name: x, self.mask_name: m})[0]
        out = out[0].transpose(1, 2, 0)
        if out.max() <= 2.0:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def self_test(self) -> bool:
        """Run a synthetic image through the model. A correct LaMa export returns the
        unmasked area (almost) unchanged; wrong channel order or value range does not.
        This protects against shipping a model whose I/O conventions differ."""
        try:
            yy, xx = np.mgrid[0:self.SIZE, 0:self.SIZE].astype(np.float32)
            img = np.stack([40 + xx / 3, 200 - yy / 3, 90 + (xx + yy) / 8], axis=2)
            img = np.clip(img, 0, 255).astype(np.uint8)
            mask = np.zeros((self.SIZE, self.SIZE), np.uint8)
            mask[200:312, 200:312] = 255
            out = self._run(img, mask)
            keep = mask == 0
            mae = float(np.abs(out.astype(np.int16) - img.astype(np.int16))[keep].mean())
            ok = mae < 10.0
            log.info("LaMa self-test: unmasked MAE %.2f -> %s", mae, "ok" if ok else "REJECTED")
            return ok
        except Exception as exc:  # noqa: BLE001
            log.warning("LaMa self-test failed (%s)", exc)
            return False

    def inpaint(self, bgr, mask, cancel: threading.Event | None = None) -> np.ndarray:
        h, w = mask.shape
        result = bgr.copy()
        for (x0, y0, x1, y1) in _clusters(mask, gap=40):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            half = max(self.SIZE // 2, int(0.75 * max(x1 - x0, y1 - y0)))
            X0, Y0 = max(0, cx - half), max(0, cy - half)
            X1, Y1 = min(w, cx + half), min(h, cy + half)
            crop, m = bgr[Y0:Y1, X0:X1], mask[Y0:Y1, X0:X1]
            filled = self._run(crop, m)
            sel = m > 0
            result[Y0:Y1, X0:X1][sel] = filled[sel]
        return result


# --------------------------------------------------------------------------
def ensure_model(path: Path, url: str, timeout: int = 60) -> bool:
    if path.exists() and path.stat().st_size > 10_000_000:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log.info("Downloading LaMa model from %s", url)
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:  # noqa: S310
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(path)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("LaMa model download failed (%s); using OpenCV fallback", exc)
        return False


def make_inpainter(use_lama: bool, model_path: Path, model_url: str):
    if use_lama:
        try:
            if ensure_model(model_path, model_url):
                inp = LamaInpainter(model_path)
                if inp.self_test():
                    log.info("Using LaMa inpainting")
                    return inp
                log.warning("LaMa model failed its self-test; using OpenCV fallback")
        except Exception as exc:  # noqa: BLE001
            log.warning("LaMa unavailable (%s); using OpenCV fallback", exc)
    log.info("Using OpenCV inpainting")
    return OpenCVInpainter()


class FallbackInpainter:
    """Try the primary engine; on any failure fall back to OpenCV."""

    def __init__(self, primary, fallback: Callable[[], OpenCVInpainter] = OpenCVInpainter):
        self.primary, self.fallback = primary, fallback()
        self.name = primary.name

    def inpaint(self, bgr, mask, cancel=None):
        if self.primary.name == "opencv":
            return self.primary.inpaint(bgr, mask, cancel)
        try:
            return self.primary.inpaint(bgr, mask, cancel)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("Primary inpainter failed (%s); falling back to OpenCV", exc)
            self.name = "opencv"
            return self.fallback.inpaint(bgr, mask, cancel)
