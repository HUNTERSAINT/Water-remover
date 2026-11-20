import io

import numpy as np
import pytest
from PIL import Image

from app.image_receiver import ImageError, find_content_box, load_image
from conftest import encode, make_photo

BIG = 20 << 20
PIX = 40_000_000


def test_rejects_empty_garbage_oversize():
    with pytest.raises(ImageError):
        load_image(b"", BIG, PIX)
    with pytest.raises(ImageError):
        load_image(b"not an image at all", BIG, PIX)
    with pytest.raises(ImageError):
        load_image(encode(make_photo()), 100, PIX)


def test_rejects_unsupported_format_and_huge_pixels():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, "BMP")
    with pytest.raises(ImageError):
        load_image(buf.getvalue(), BIG, PIX)
    with pytest.raises(ImageError):
        load_image(encode(make_photo()), BIG, 1000)


def test_truncated_file_is_handled():
    data = encode(make_photo())
    with pytest.raises(ImageError):
        load_image(data[: len(data) // 3], BIG, PIX)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_accepts_formats(fmt):
    li = load_image(encode(make_photo(), fmt), BIG, PIX)
    assert li.fmt == fmt and li.size == (640, 480)


def test_crops_flat_black_screenshot_bars():
    photo = make_photo(400, 500)
    canvas = np.zeros((900, 400, 3), np.uint8)
    canvas[200:700] = photo
    box = find_content_box(canvas)
    assert box is not None
    x0, y0, x1, y1 = box
    assert abs(y0 - 200) <= 3 and abs(y1 - 700) <= 3


def test_does_not_crop_dark_natural_photo():
    rng = np.random.default_rng(0)
    dark = np.clip(rng.normal(8, 6, (400, 400, 3)), 0, 255).astype(np.uint8)
    dark[150:250, 100:300] = 120
    assert find_content_box(dark) is None


def test_alpha_is_kept():
    rgba = np.zeros((200, 200, 4), np.uint8)
    rgba[..., :3] = 150
    rgba[:, :100, 3] = 255
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    li = load_image(buf.getvalue(), BIG, PIX, auto_crop=False)
    assert li.alpha is not None and li.alpha[:, 150].max() == 0
