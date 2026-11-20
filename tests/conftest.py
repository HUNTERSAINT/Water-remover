import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_photo(w=640, h=480, seed=3):
    """A plain-ish product-like photo with gentle texture and a crisp printed label."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 190 + 25 * np.sin(xx / 90.0) + 15 * np.cos(yy / 70.0)
    img = np.stack([base, base * 0.97, base * 0.94], axis=2)
    img += rng.normal(0, 2.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # crisp product label with serial number (must NOT be removed)
    cv2.rectangle(img, (40, 40), (300, 110), (40, 40, 160), -1)
    cv2.putText(img, "SERIAL 48213", (52, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def add_watermark(clean, text="BOLT WORLD", alpha=0.35, scale=2.0, y=0.62):
    out = clean.copy().astype(np.float32)
    layer = np.zeros(clean.shape[:2], np.uint8)
    h, w = clean.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
    cv2.putText(layer, text, ((w - tw) // 2, int(h * y)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 4, cv2.LINE_AA)
    a = (layer.astype(np.float32) / 255.0 * alpha)[:, :, None]
    out = out * (1 - a) + np.array([110, 110, 110], np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8), layer > 0


def encode(bgr, fmt="JPEG", **kw):
    buf = io.BytesIO()
    Image.fromarray(bgr[:, :, ::-1]).save(buf, fmt, **kw)
    return buf.getvalue()


@pytest.fixture
def clean():
    return make_photo()


@pytest.fixture
def watermarked(clean):
    wm, layer = add_watermark(clean)
    return wm, layer
