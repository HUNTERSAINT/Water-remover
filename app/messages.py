WELCOME = (
    "👋 Welcome to AI Image Cleaner!\n\n"
    "Send me an image and I'll help remove unwanted watermarks while preserving "
    "the original image quality.\n\n"
    "✨ Features:\n"
    "• Automatic watermark detection\n"
    "• Text watermark removal\n"
    "• Logo and overlay removal\n"
    "• Background reconstruction\n"
    "• High-quality image output\n\n"
    "📸 Simply upload your image to get started."
)

HELP = (
    "ℹ️ How to use AI Image Cleaner\n\n"
    "1. Send a JPG, PNG or WEBP image. For best quality send it as a *file* "
    "(📎 → File) so Telegram does not compress it.\n"
    "2. I scan it for text watermarks and overlays.\n"
    "3. I remove them and send you the cleaned image back.\n\n"
    "If I'm not sure about something I'll show you a numbered preview so you can "
    "choose what to remove, or mark an area yourself.\n\n"
    "Commands:\n"
    "/start - Welcome message\n"
    "/help - This help\n"
    "/about - About the bot\n"
    "/cancel - Cancel the current job\n\n"
    "⚠️ Only use this on images you own or have permission to edit."
)

ABOUT = (
    "🤖 AI Image Cleaner\n\n"
    "Detects watermark text with an open-source OCR model, builds a tight mask "
    "around only the letters, and rebuilds the background with inpainting "
    "(LaMa when available, OpenCV otherwise). Everything runs on free open-source "
    "software. Your images are processed in memory and never stored.\n\n"
    "Limits: heavy watermarks over detailed or edge-rich areas can leave soft "
    "patches, and logos that are not text may need the manual area tool.\n\n"
    "Only process images you own or have permission to edit."
)

RECEIVED = "📥 Image received. Analysing your image..."
PROCESSING = "⚙️ Processing your image..."
