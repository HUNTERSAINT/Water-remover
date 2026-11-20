import numpy as np

from app import image_optimizer, mask_generator, pipeline, watermark_detector as wd
from app.config import load_settings
from app.image_inpainting import OpenCVInpainter
from app.image_receiver import load_image
from conftest import add_watermark, encode


def test_detects_watermark_and_protects_serial(watermarked):
    wm, _ = watermarked
    det = wd.detect(wm)
    texts = {r.text.upper().replace(" ", ""): r.kind for r in det.regions}
    assert any("BOLT" in t and k == wd.WATERMARK for t, k in texts.items()), texts
    serial = [r for r in det.regions if "SERIAL" in r.text.upper() or "48213" in r.text]
    assert serial and all(r.kind == wd.PROTECTED for r in serial), texts


def test_clean_image_has_no_watermark(clean):
    det = wd.detect(clean)
    assert det.watermarks == []


def test_removal_improves_masked_area_and_leaves_rest_untouched(clean, watermarked):
    wm, layer = watermarked
    det = wd.detect(wm)
    mask = mask_generator.build_mask(wm, det.watermarks)
    assert mask.any()
    out = OpenCVInpainter().inpaint(wm, mask)
    out[mask == 0] = wm[mask == 0]

    rep = image_optimizer.compare(wm, out, mask)
    assert rep["outside_max_diff"] == 0
    # label area must be bit-identical
    assert np.array_equal(out[40:110, 40:300], wm[40:110, 40:300])

    err_before = np.abs(wm.astype(int) - clean.astype(int))[layer].mean()
    err_after = np.abs(out.astype(int) - clean.astype(int))[layer].mean()
    assert err_after < 0.35 * err_before, (err_before, err_after)


def test_full_pipeline_roundtrip_keeps_dimensions_and_format(watermarked):
    wm, _ = watermarked
    cfg = load_settings(require_token=False)
    for fmt in ("JPEG", "PNG", "WEBP"):
        an = pipeline.analyse(encode(wm, fmt), cfg)
        out = pipeline.remove(an, pipeline.Options(), OpenCVInpainter(), cfg)
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(out.data))
        assert im.format == fmt and im.size == (640, 480)
        assert out.report["outside_max_diff"] == 0


def test_manual_box_removes_something(watermarked):
    wm, layer = watermarked
    cfg = load_settings(require_token=False)
    an = pipeline.analyse(encode(wm, "PNG"), cfg)
    ys, xs = np.where(layer)
    box = (int(xs.min()) - 10, int(ys.min()) - 10, int(xs.max()) + 10, int(ys.max()) + 10)
    out = pipeline.remove(an, pipeline.Options(selected=[], manual_boxes=[box]), OpenCVInpainter(), cfg)
    assert out.report["mask_fraction"] > 0


def test_previews_render(watermarked):
    wm, _ = watermarked
    cfg = load_settings(require_token=False)
    an = pipeline.analyse(encode(wm), cfg)
    assert pipeline.annotated_preview(an)[:2] == b"\xff\xd8"
    assert pipeline.grid_preview(an, {1, 5})[:2] == b"\xff\xd8"
