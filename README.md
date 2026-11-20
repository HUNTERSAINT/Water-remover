# AI Image Cleaner - Telegram watermark remover

Send the bot a picture; it finds text watermarks, removes only the letters, rebuilds the
background and sends the cleaned file back. 100% free and open source: no paid APIs, no
API keys other than your Telegram bot token.

## How it works

| Stage | File | What it does |
|---|---|---|
| Receive | `image_receiver.py` | Validates type/size/pixels, decodes safely, fixes EXIF rotation, auto-crops flat black screenshot bars (e.g. Jiji/gallery screenshots) |
| Detect | `watermark_detector.py` | RapidOCR (PaddleOCR models on onnxruntime, CPU). A second pass on a contrast-boosted copy finds faint overlays. Each text box is scored: faint strokes, watermark wording (jiji, posted on, www, ©...), repeated text, overlay-sized text → **watermark / uncertain / protected** |
| Mask | `mask_generator.py` | Follows only the letter strokes (top-hat/black-hat), grown a few pixels. Not a rectangle |
| Inpaint | `image_inpainting.py` | LaMa (optional) or OpenCV Telea+Navier-Stokes with grain re-injection. Auto-fallback to OpenCV on any failure |
| Verify/encode | `image_optimizer.py` | Confirms nothing outside the mask changed, estimates leftover traces, re-encodes in the original format at high quality |
| Deliver | `image_sender.py` | Private temp folder, sent as a *document* (no Telegram recompression), then deleted |

**Protected text:** crisp, high-contrast print (product labels, serial numbers, the 1/2 · 1/4 measurement
numbers) is never removed automatically. Borderline text is *not touched*; the bot shows a numbered
preview and asks you to confirm. If nothing is found you can pick the area on a 4x4 grid.

## Bot commands
`/start` `/help` `/about` `/cancel`. After each result: **Retry (stronger)**, **Review text areas**, **Done**.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # put your token from @BotFather in TELEGRAM_BOT_TOKEN
export $(grep -v '^#' .env | xargs)
python main.py
```

## Deploy on your existing host
Uses Telegram **long polling**, so no public URL or webhook is needed.

* **Railway (or any Docker host):** push this folder to a GitHub repo, create a service from it
  (the `Dockerfile` is detected automatically), add the variable `TELEGRAM_BOT_TOKEN`. That's all.
* **Heroku-style hosts (Procfile):** `worker: python main.py`. These images need `libgl1` and
  `libglib2.0-0` installed (use the Dockerfile if unsure).
* **A VPS:** `docker build -t image-cleaner . && docker run -d --restart=always -e TELEGRAM_BOT_TOKEN=... image-cleaner`
* If your host injects `PORT`, the bot serves `GET /` → `ok` for health checks.

Run **only one instance** per bot token (two pollers conflict).

### Resources
* Default (OpenCV), measured: ~315 MB RAM peak and ~2 s for a 720x958 image; ~650 MB peak and ~8 s for a
  12-megapixel photo (2 CPU cores). On small plans lower `MAX_MEGAPIXELS` and `MAX_CONCURRENT_JOBS`.
* `USE_LAMA=true`: much better on watermarks crossing edges, but expects 1.5 GB+ RAM and downloads the
  model on first start. If the model can't load, the bot silently uses OpenCV.

## Environment variables
See `.env.example`. Only `TELEGRAM_BOT_TOKEN` is required. Secrets are never logged (a filter redacts tokens).

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
27 tests: input validation (corrupt/oversized/wrong type), crop logic, detection (watermark found,
serial-number label protected), removal quality (error inside the mask drops by >65%, bytes outside the
mask identical), all three formats and dimensions, LaMa plumbing with a stand-in model, OpenCV fallback,
and the full Telegram flow with a mocked transport (receive → analyse → process → document returned,
retry, cancel, timeout, busy user, one user cannot use another's session, temp files deleted).

## Limitations (please read)
* Inpainting **invents** the pixels under the letters. On smooth areas it is nearly invisible; where a
  watermark crosses sharp edges, fine text or skin/fabric texture you may see soft patches (OpenCV) or
  small artefacts (LaMa). It cannot recover detail that was truly hidden.
* Detection is **text-based**. Logos and image-only marks are not auto-detected; use the manual grid.
  Very faint or heavily stylised text can be missed (use Mark area) or flagged "unsure".
* Printed text with low contrast could be classed as a watermark; borderline cases go to the confirm step.
* Photos sent as photos are already JPEG-compressed by Telegram; send as **File** for best quality.
* Telegram bots can download files up to 20 MB (limit here: `MAX_IMAGE_MB`).
* Use only on images you own or have permission to edit.

## Privacy
Images are processed in memory; the only disk use is a private temp file for the reply, deleted
immediately. Nothing is stored, and sessions expire after 15 minutes.
