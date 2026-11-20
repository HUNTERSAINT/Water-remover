"""Telegram handlers. Long polling by default."""
from __future__ import annotations

import asyncio
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    Application, ApplicationBuilder, CallbackQueryHandler, CommandHandler,
    ContextTypes, MessageHandler, filters,
)

from . import image_receiver, image_sender, messages, pipeline
from .config import Settings, load_settings
from .image_inpainting import Cancelled, FallbackInpainter, make_inpainter
from .logging_setup import setup_logging
from .watermark_detector import PROTECTED, WATERMARK

log = logging.getLogger(__name__)


class JobTimeout(Exception):
    pass


@dataclass
class Session:
    sid: str
    user_id: int
    analysis: pipeline.Analysis
    created: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)
    busy: bool = False
    selected: set[int] = field(default_factory=set)      # region numbers (1-based)
    cells: set[int] = field(default_factory=set)         # manual grid cells (1-based)


# ----------------------------------------------------------------- helpers
def _state(ctx: ContextTypes.DEFAULT_TYPE) -> dict:
    return ctx.application.bot_data


def _sessions(ctx) -> dict[int, Session]:
    return _state(ctx)["sessions"]


def _drop(ctx, user_id: int) -> None:
    s = _sessions(ctx).pop(user_id, None)
    if s:
        s.cancel.set()


async def _run(ctx, session: Session, fn, *args):
    cfg: Settings = _state(ctx)["cfg"]
    async with _state(ctx)["sem"]:
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), cfg.job_timeout_s)
        except asyncio.TimeoutError as exc:
            session.cancel.set()
            raise JobTimeout() from exc


def _region_keyboard(session: Session) -> InlineKeyboardMarkup:
    rows, row = [], []
    for i, r in enumerate(session.analysis.detection.regions, 1):
        if r.kind == PROTECTED:
            continue
        mark = "✅" if i in session.selected else "⬜"
        row.append(InlineKeyboardButton(f"{mark} {i} {r.text[:14]}", callback_data=f"t:{session.sid}:{i}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([
        InlineKeyboardButton("🧹 Remove selected", callback_data=f"go:{session.sid}"),
        InlineKeyboardButton("✏️ Mark area", callback_data=f"m:{session.sid}"),
    ])
    rows.append([InlineKeyboardButton("❌ Cancel", callback_data=f"x:{session.sid}")])
    return InlineKeyboardMarkup(rows)


def _grid_keyboard(session: Session, cols: int = 4, rows_n: int = 4) -> InlineKeyboardMarkup:
    rows = []
    for r in range(rows_n):
        rows.append([
            InlineKeyboardButton(
                ("🟩" if (r * cols + c + 1) in session.cells else "⬜") + str(r * cols + c + 1),
                callback_data=f"c:{session.sid}:{r * cols + c + 1}",
            ) for c in range(cols)
        ])
    rows.append([
        InlineKeyboardButton("🧹 Remove marked area", callback_data=f"mg:{session.sid}"),
        InlineKeyboardButton("❌ Cancel", callback_data=f"x:{session.sid}"),
    ])
    return InlineKeyboardMarkup(rows)


def _result_keyboard(session: Session) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔁 Retry (stronger)", callback_data=f"retry:{session.sid}"),
        InlineKeyboardButton("🔎 Review text areas", callback_data=f"rv:{session.sid}"),
        InlineKeyboardButton("✅ Done", callback_data=f"x:{session.sid}"),
    ]])


def _caption(outcome: pipeline.Outcome, left_alone: int) -> str:
    lines = [f"✅ Done! Removed {outcome.regions_removed} watermark area(s)."]
    if outcome.traces_may_remain:
        lines.append("⚠️ Faint traces may remain. Tap Retry (stronger) if you can see them.")
    if left_alone:
        lines.append(f"ℹ️ I left {left_alone} uncertain text area(s) untouched. Tap Review to choose.")
    lines.append("Everything outside the watermark was kept exactly as it was.")
    return "\n".join(lines)


# ---------------------------------------------------------------- commands
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(messages.WELCOME)


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(messages.HELP, parse_mode="Markdown")


async def cmd_about(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(messages.ABOUT)


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    had = uid in _sessions(ctx)
    _drop(ctx, uid)
    await update.effective_message.reply_text(
        "🛑 Cancelled. Your image has been discarded." if had else "Nothing to cancel."
    )


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("📸 Please send me an image (JPG, PNG or WEBP).")


async def on_bad_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("That doesn't look like an image. Please send a JPG, PNG or WEBP.")


# ------------------------------------------------------------ main workflow
async def on_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, user = update.effective_message, update.effective_user
    cfg: Settings = _state(ctx)["cfg"]

    old = _sessions(ctx).get(user.id)
    if old and old.busy:
        await msg.reply_text("⏳ I'm still working on your previous image. Send /cancel to stop it.")
        return
    _drop(ctx, user.id)

    tg_file = msg.photo[-1] if msg.photo else msg.document
    if tg_file.file_size and tg_file.file_size > cfg.max_image_bytes:
        await msg.reply_text(f"⚠️ That file is too large. The limit is {cfg.max_image_mb} MB.")
        return

    status = await msg.reply_text(messages.RECEIVED)
    placeholder = Session(secrets.token_urlsafe(6), user.id, None)  # type: ignore[arg-type]
    placeholder.busy = True
    _sessions(ctx)[user.id] = placeholder
    try:
        await ctx.bot.send_chat_action(msg.chat_id, ChatAction.UPLOAD_PHOTO)
        data = bytes(await (await tg_file.get_file()).download_as_bytearray())
        analysis = await _run(ctx, placeholder, pipeline.analyse, data, cfg, placeholder.cancel)
        del data
    except image_receiver.ImageError as exc:
        _drop(ctx, user.id)
        await status.edit_text(f"⚠️ {exc}")
        return
    except Cancelled:
        await status.edit_text("🛑 Cancelled.")
        return
    except JobTimeout:
        _drop(ctx, user.id)
        await status.edit_text("⏱️ That took too long and was stopped. Try a smaller image.")
        return
    except Exception:  # noqa: BLE001
        log.exception("Analysis failed")
        _drop(ctx, user.id)
        await status.edit_text("❌ Something went wrong while analysing that image. Please try again.")
        return

    if _sessions(ctx).get(user.id) is not placeholder:      # cancelled meanwhile
        return
    session = placeholder
    session.analysis = analysis
    session.busy = False
    det = analysis.detection
    session.selected = {i for i, r in enumerate(det.regions, 1) if r.kind != PROTECTED}

    if det.watermarks:
        await _process(update, ctx, session, status, pipeline.Options())
    elif det.uncertain:
        await status.edit_text("🤔 I found some text but I'm not sure it's a watermark. Nothing was changed yet.")
        await _send_review(msg, ctx, session)
    else:
        await status.edit_text(
            "🔍 I couldn't find any text watermark. If there's a logo or mark, "
            "tap Mark area and select where it is.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("✏️ Mark area", callback_data=f"m:{session.sid}"),
                InlineKeyboardButton("❌ Cancel", callback_data=f"x:{session.sid}"),
            ]]),
        )


async def _send_review(msg, ctx, session: Session):
    preview = pipeline.annotated_preview(session.analysis, session.selected)
    await msg.reply_photo(
        preview,
        caption=("Green = likely watermark, orange = unsure, grey = unselected.\n"
                 "Tick what to remove, then tap Remove selected."),
        reply_markup=_region_keyboard(session),
    )


async def _process(update, ctx, session: Session, status, options: pipeline.Options):
    """Run removal and deliver the result. `status` is a message to edit."""
    cfg: Settings = _state(ctx)["cfg"]
    session.busy = True
    session.cancel.clear()
    try:
        await status.edit_text(messages.PROCESSING)
        outcome = await _run(
            ctx, session, pipeline.remove, session.analysis, options,
            _state(ctx)["inpainter"], cfg, session.cancel,
        )
        left = len([r for r in session.analysis.detection.uncertain
                    if options.selected is None and not options.include_uncertain])
        await image_sender.send_result(
            status, cfg.temp_root, outcome.data, outcome.ext,
            _caption(outcome, left), _result_keyboard(session),
        )
        await status.delete()
    except Cancelled:
        try:
            await status.edit_text("🛑 Cancelled.")
        except TelegramError:
            pass
    except JobTimeout:
        await status.edit_text("⏱️ Processing took too long and was stopped.")
    except ValueError:
        await status.edit_text("⚠️ There was nothing selected to remove.")
    except Exception:  # noqa: BLE001
        log.exception("Removal failed")
        try:
            await status.edit_text("❌ Something went wrong while processing. Please try again.")
        except TelegramError:
            pass
    finally:
        session.busy = False


# --------------------------------------------------------------- callbacks
async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    parts = (q.data or "").split(":")
    action, sid = parts[0], (parts[1] if len(parts) > 1 else "")
    session = _sessions(ctx).get(q.from_user.id)
    if session is None or session.sid != sid or session.analysis is None:
        await q.answer("This session has expired. Please send the image again.", show_alert=True)
        return
    if session.busy and action != "x":
        await q.answer("Still working…")
        return

    if action == "x":
        _drop(ctx, q.from_user.id)
        await q.answer("Discarded")
        try:
            await q.edit_message_reply_markup(None)
        except TelegramError:
            pass
        return

    if action == "t":                                   # toggle a region
        i = int(parts[2])
        session.selected ^= {i}
        await q.answer()
        await q.edit_message_reply_markup(_region_keyboard(session))
        return

    if action == "rv":
        await q.answer()
        await _send_review(q.message, ctx, session)
        return

    if action == "m":                                   # manual area picker
        session.cells.clear()
        await q.answer()
        await q.message.reply_photo(
            pipeline.grid_preview(session.analysis, session.cells),
            caption="✏️ Tap the squares that cover the watermark (pick as few as possible), then Remove.",
            reply_markup=_grid_keyboard(session),
        )
        return

    if action == "c":
        session.cells ^= {int(parts[2])}
        await q.answer()
        await q.edit_message_reply_markup(_grid_keyboard(session))
        return

    status = await q.message.reply_text(messages.PROCESSING)
    await q.answer()
    cfg: Settings = _state(ctx)["cfg"]

    if action == "go":
        det = session.analysis.detection
        chosen = [r for i, r in enumerate(det.regions, 1) if i in session.selected and r.kind != PROTECTED]
        await _process(update, ctx, session, status, pipeline.Options(selected=chosen))
    elif action == "mg":
        if not session.cells:
            await status.edit_text("Please tap at least one square first.")
            return
        boxes = pipeline.grid_boxes(session.analysis.loaded.size)
        await _process(update, ctx, session, status, pipeline.Options(
            selected=[], manual_boxes=[boxes[i - 1] for i in sorted(session.cells)]))
    elif action == "retry":
        await _process(update, ctx, session, status,
                       pipeline.Options(strength=1.8, include_uncertain=True))


async def on_error(update, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("Unhandled error: %s", ctx.error, exc_info=ctx.error)


# -------------------------------------------------------------- app wiring
async def _sweeper(app: Application):
    cfg: Settings = app.bot_data["cfg"]
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for uid, s in list(app.bot_data["sessions"].items()):
            if not s.busy and now - s.created > cfg.session_ttl_s:
                app.bot_data["sessions"].pop(uid, None)
        await asyncio.to_thread(image_sender.prepare_root, cfg.temp_root)


async def _post_init(app: Application):
    cfg: Settings = app.bot_data["cfg"]
    image_sender.prepare_root(cfg.temp_root)
    primary = await asyncio.to_thread(
        make_inpainter, cfg.use_lama, cfg.lama_model_path, cfg.lama_model_url)
    app.bot_data["inpainter"] = FallbackInpainter(primary)
    await asyncio.to_thread(_warm_ocr)
    app.bot_data["sweeper"] = asyncio.create_task(_sweeper(app))
    log.info("Bot ready (engine: %s)", app.bot_data["inpainter"].name)


def _warm_ocr():
    from . import watermark_detector
    watermark_detector._get_engine()


def _start_health_server(port: int):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def log_message(self, *a):  # silence
            pass

    srv = HTTPServer(("0.0.0.0", port), H)  # noqa: S104
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("Health endpoint listening on :%d", port)


def build_application(cfg: Settings) -> Application:
    app = (ApplicationBuilder().token(cfg.token).concurrent_updates(True)
           .post_init(_post_init).build())
    app.bot_data.update(cfg=cfg, sessions={}, sem=asyncio.Semaphore(cfg.max_concurrent_jobs))
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_image))
    app.add_handler(MessageHandler(filters.Document.ALL, on_bad_document))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    return app


def main():
    cfg = load_settings()
    setup_logging(cfg.log_level)
    if cfg.health_port:
        _start_health_server(cfg.health_port)
    app = build_application(cfg)
    log.info("Starting long polling")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
