"""Checks the LaMa code path (crop, 512 resize, paste-back) using a tiny stand-in
ONNX model that fills the mask with mid-grey. The real LaMa weights are not
needed to prove the plumbing is correct."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from app.image_inpainting import FallbackInpainter, LamaInpainter


def _fake_model(path):
    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 512, 512])
    msk = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 512, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 512, 512])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    half = helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5])
    s255 = helper.make_tensor("s255", TensorProto.FLOAT, [1], [255.0])
    nodes = [
        helper.make_node("Sub", ["one", "mask"], ["inv"]),
        helper.make_node("Mul", ["image", "inv"], ["keep"]),
        helper.make_node("Mul", ["mask", "half"], ["fill"]),
        helper.make_node("Add", ["keep", "fill"], ["sum"]),
        helper.make_node("Mul", ["sum", "s255"], ["output"]),
    ]
    g = helper.make_graph(nodes, "fake", [img, msk], [out], initializer=[one, half, s255])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_lama_path_fills_only_masked_pixels(tmp_path):
    p = tmp_path / "fake.onnx"
    _fake_model(p)
    inp = LamaInpainter(p)
    img = np.full((700, 900, 3), 200, np.uint8)
    mask = np.zeros((700, 900), np.uint8)
    mask[300:340, 300:600] = 255
    out = inp.inpaint(img, mask)
    assert abs(int(out[320, 450, 0]) - 127) <= 2          # filled
    assert np.array_equal(out[mask == 0], img[mask == 0])  # rest identical


def test_fallback_to_opencv_when_primary_crashes():
    class Boom:
        name = "lama"
        def inpaint(self, *a, **k):
            raise RuntimeError("model exploded")
    img = np.full((200, 200, 3), 180, np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    mask[90:110, 50:150] = 255
    fb = FallbackInpainter(Boom())
    out = fb.inpaint(img, mask)
    assert fb.name == "opencv" and out.shape == img.shape
