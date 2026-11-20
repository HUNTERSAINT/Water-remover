"""Temporary workspace and delivery of results back to Telegram."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


def prepare_root(root: Path, max_age_s: int = 3600) -> None:
    """Create the temp root (private) and sweep leftovers from crashed runs."""
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    now = time.time()
    for child in root.iterdir():
        try:
            if now - child.stat().st_mtime > max_age_s:
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
        except OSError:
            pass


@contextmanager
def job_dir(root: Path):
    """A private per-job folder that is always deleted afterwards."""
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="job-", dir=root))
    os.chmod(path, 0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


async def send_result(message, root: Path, data: bytes, ext: str, caption: str, reply_markup=None):
    """Write the cleaned image to a private temp file, send it as a document
    (Telegram does not recompress documents, so quality is preserved), then
    delete the file."""
    with job_dir(root) as d:
        out = d / f"cleaned.{ext}"
        out.write_bytes(data)
        with open(out, "rb") as fh:
            return await message.reply_document(
                document=fh, filename=f"cleaned.{ext}", caption=caption,
                reply_markup=reply_markup,
            )
