"""Drives the real handlers with a mocked Telegram transport."""
import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app import bot, messages
from app.config import load_settings
from app.image_inpainting import FallbackInpainter, OpenCVInpainter
from conftest import encode, make_photo


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "tmp"))
    cfg = load_settings(require_token=False)
    data = dict(cfg=cfg, sessions={}, sem=asyncio.Semaphore(2),
                inpainter=FallbackInpainter(OpenCVInpainter()))
    return SimpleNamespace(application=SimpleNamespace(bot_data=data), bot=AsyncMock()), cfg


def make_update(img_bytes, user_id=1, size=None):
    sent = {}
    status = AsyncMock()

    async def reply_document(document, filename, caption, reply_markup=None):
        sent["bytes"] = document.read()
        sent["filename"], sent["caption"], sent["markup"] = filename, caption, reply_markup

    status.reply_document = AsyncMock(side_effect=reply_document)
    tg_file = MagicMock()
    tg_file.file_size = size if size is not None else len(img_bytes)
    tg_file.get_file = AsyncMock(return_value=SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=bytearray(img_bytes))))
    msg = MagicMock()
    msg.photo, msg.document, msg.chat_id = [tg_file], None, 99
    msg.reply_text = AsyncMock(return_value=status)
    msg.reply_photo = AsyncMock()
    upd = SimpleNamespace(effective_message=msg, effective_user=SimpleNamespace(id=user_id))
    return upd, msg, status, sent


async def test_full_flow_watermark_removed_and_returned(ctx, watermarked, tmp_path):
    ctx, cfg = ctx
    wm, _ = watermarked
    upd, msg, status, sent = make_update(encode(wm, "PNG"))

    await bot.on_image(upd, ctx)

    assert msg.reply_text.call_args_list[0].args[0] == messages.RECEIVED
    texts = [c.args[0] for c in status.edit_text.call_args_list]
    assert messages.PROCESSING in texts
    assert sent["filename"] == "cleaned.png" and "Done" in sent["caption"]
    out = Image.open(io.BytesIO(sent["bytes"]))
    assert out.size == (640, 480)
    # retry option is offered
    assert any("Retry" in b.text for row in sent["markup"].inline_keyboard for b in row)
    # temp files are gone
    assert not [p for p in (tmp_path / "tmp").glob("job-*")]
    # session kept for retry, not busy
    s = ctx.application.bot_data["sessions"][1]
    assert s.busy is False


async def test_retry_button_reprocesses(ctx, watermarked):
    ctx, cfg = ctx
    wm, _ = watermarked
    upd, msg, status, sent = make_update(encode(wm, "JPEG"))
    await bot.on_image(upd, ctx)
    sid = ctx.application.bot_data["sessions"][1].sid

    status2 = AsyncMock(); sent2 = {}
    async def rd(document, filename, caption, reply_markup=None):
        sent2["bytes"] = document.read()
    status2.reply_document = AsyncMock(side_effect=rd)
    q = MagicMock(); q.data = f"retry:{sid}"; q.from_user = SimpleNamespace(id=1)
    q.answer = AsyncMock(); q.message.reply_text = AsyncMock(return_value=status2)
    await bot.on_callback(SimpleNamespace(callback_query=q), ctx)
    assert Image.open(io.BytesIO(sent2["bytes"])).format == "JPEG"


async def test_other_users_cannot_use_a_session(ctx, watermarked):
    ctx, cfg = ctx
    wm, _ = watermarked
    upd, *_ = make_update(encode(wm))
    await bot.on_image(upd, ctx)
    sid = ctx.application.bot_data["sessions"][1].sid
    q = MagicMock(); q.data = f"retry:{sid}"; q.from_user = SimpleNamespace(id=2)
    q.answer = AsyncMock(); q.message.reply_text = AsyncMock()
    await bot.on_callback(SimpleNamespace(callback_query=q), ctx)
    assert "expired" in q.answer.call_args.args[0]
    q.message.reply_text.assert_not_called()


async def test_no_watermark_offers_manual_area(ctx, clean):
    ctx, cfg = ctx
    upd, msg, status, sent = make_update(encode(clean))
    await bot.on_image(upd, ctx)
    last = status.edit_text.call_args_list[-1]
    assert "couldn't find" in last.args[0]
    assert "Mark area" in str(last.kwargs["reply_markup"])
    assert "bytes" not in sent


async def test_corrupt_and_oversized_files_rejected_politely(ctx):
    ctx, cfg = ctx
    upd, msg, status, sent = make_update(b"garbage-bytes" * 50)
    await bot.on_image(upd, ctx)
    assert "couldn't read" in status.edit_text.call_args.args[0]
    assert 1 not in ctx.application.bot_data["sessions"]

    upd, msg, status, sent = make_update(b"x", size=cfg.max_image_bytes + 1)
    await bot.on_image(upd, ctx)
    assert "too large" in msg.reply_text.call_args.args[0]


async def test_busy_user_is_told_to_wait_and_cancel_works(ctx, watermarked):
    ctx, cfg = ctx
    wm, _ = watermarked
    upd, *_ = make_update(encode(wm))
    await bot.on_image(upd, ctx)
    ctx.application.bot_data["sessions"][1].busy = True
    upd2, msg2, *_ = make_update(encode(wm))
    await bot.on_image(upd2, ctx)
    assert "still working" in msg2.reply_text.call_args.args[0]

    cmsg = MagicMock(); cmsg.reply_text = AsyncMock()
    await bot.cmd_cancel(SimpleNamespace(effective_message=cmsg, effective_user=SimpleNamespace(id=1)), ctx)
    assert 1 not in ctx.application.bot_data["sessions"]


async def test_timeout_is_reported(ctx, watermarked, monkeypatch):
    ctx, cfg = ctx
    wm, _ = watermarked
    import dataclasses, time
    ctx.application.bot_data["cfg"] = dataclasses.replace(cfg, job_timeout_s=1)
    monkeypatch.setattr(bot.pipeline, "analyse", lambda *a, **k: time.sleep(3))
    upd, msg, status, sent = make_update(encode(wm))
    await bot.on_image(upd, ctx)
    assert "too long" in status.edit_text.call_args.args[0]
