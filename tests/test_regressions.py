"""Regressions for bugs found in the live deployment."""
import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from telegram.error import BadRequest

from app import bot, mask_generator, messages, pipeline, watermark_detector as wd
from app.config import load_settings
from app.image_inpainting import FallbackInpainter, OpenCVInpainter
from conftest import encode


def test_small_faint_label_text_is_not_auto_removed():
    """A faint-but-small product label word ("TIMER") must never be auto-deleted."""
    img = np.full((506, 689, 3), 150, np.uint8)
    r = wd.TextRegion((66, 357, 96, 376), np.array([[66, 357], [96, 357], [96, 376], [66, 376]], np.int32),
                      "TIMER", 0.9, contrast=35.0)
    wd.classify([r], cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert r.kind != wd.WATERMARK


def test_overlay_sized_faint_line_is_a_watermark():
    img = np.full((506, 689, 3), 150, np.uint8)
    r = wd.TextRegion((166, 313, 503, 336), np.array([[166, 313], [503, 313], [503, 336], [166, 336]], np.int32),
                      "MOTEADEZ ELECTRONICS", 0.8, contrast=44.0)
    wd.classify([r], cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert r.kind == wd.WATERMARK


def test_fragments_on_one_line_are_merged():
    def reg(x0, x1):
        return wd.TextRegion((x0, 313, x1, 336), np.array([[x0, 313], [x1, 313], [x1, 336], [x0, 336]], np.int32),
                             "frag", 0.7, contrast=44.0, enhanced_only=True)
    merged = wd._merge_lines([reg(166, 260), reg(253, 420), reg(330, 503)])
    assert len(merged) == 1 and merged[0].box[0] == 166 and merged[0].box[2] == 503


def test_manual_area_without_text_never_wipes_the_rectangle():
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(140, 10, (400, 600, 3)), 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = (100, 100, 400, 300)
    mask = mask_generator.region_mask(gray, box=box)
    area = (box[2] - box[0]) * (box[3] - box[1])
    assert (mask[100:300, 100:400] > 0).sum() < 0.25 * area   # only the strongest marks, not everything
    assert (mask > 0).sum() < 0.30 * mask.size


def test_grid_is_fine_enough_to_pick_a_text_line():
    assert len(pipeline.grid_boxes((600, 400))) == 36


# ---- the "Message is not modified" crash seen on Railway --------------------
class StrictStatus:
    """Behaves like Telegram: editing to identical text raises BadRequest."""
    def __init__(self, text):
        self.text = text
        self.sent = {}

    async def edit_text(self, text, **kw):
        if text == self.text:
            raise BadRequest("Message is not modified: specified new message content and reply markup are exactly the same")
        self.text = text

    async def reply_document(self, document, filename, caption, reply_markup=None):
        self.sent["bytes"] = document.read()

    async def delete(self):
        pass


async def test_callback_status_message_does_not_crash_on_identical_text(tmp_path, monkeypatch, watermarked):
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "tmp"))
    cfg = load_settings(require_token=False)
    ctx = SimpleNamespace(application=SimpleNamespace(bot_data=dict(
        cfg=cfg, sessions={}, sem=asyncio.Semaphore(2), inpainter=FallbackInpainter(OpenCVInpainter()))),
        bot=AsyncMock())
    wm, _ = watermarked
    an = pipeline.analyse(encode(wm, "PNG"), cfg)
    s = bot.Session("abc", 1, an)
    s.selected = {i for i, r in enumerate(an.detection.regions, 1) if r.kind != wd.PROTECTED}
    ctx.application.bot_data["sessions"][1] = s

    status = StrictStatus(messages.PROCESSING)       # same text _process will set
    q = MagicMock(); q.data = "retry:abc"; q.from_user = SimpleNamespace(id=1)
    q.answer = AsyncMock(); q.message.reply_text = AsyncMock(return_value=status)
    await bot.on_callback(SimpleNamespace(callback_query=q), ctx)
    assert "bytes" in status.sent                     # result was delivered despite the identical edit
