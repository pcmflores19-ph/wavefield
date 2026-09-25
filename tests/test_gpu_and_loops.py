"""
The GPU encoder argument builder, and the numba-compiled effect loops against
plain-Python references (numba is optional, so both paths must agree).
"""
import numpy as np
import pytest

import effects
import video_export


@pytest.mark.parametrize("encoder, expected", [
    ("h264_nvenc", ["-rc", "vbr", "-cq", "20"]),
    ("h264_amf", ["-rc", "cqp", "-qp_i", "20"]),
    ("h264_qsv", ["-global_quality", "20"]),
])
def test_gpu_video_codec_arguments(encoder, expected):
    args = video_export.gpu_video_codec(encoder, 20)
    assert args[:2] == ["-c:v", encoder]
    joined = " ".join(args)
    assert " ".join(expected) in joined


def test_gpu_video_codec_rejects_unknown_encoder():
    with pytest.raises(ValueError):
        video_export.gpu_video_codec("h264_bogus", 20)


def _py(fn):
    """The un-jitted function, whether or not numba wrapped it."""
    return getattr(fn, "py_func", fn)


def test_one_pole_loop_matches_reference():
    x = np.random.default_rng(1).standard_normal(2000).astype(np.float32)
    a = 0.9
    out, s = effects._one_pole_loop(x, a, 0.25)
    ref, s_ref = _py(effects._one_pole_loop)(x, a, 0.25)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
    assert s == pytest.approx(s_ref, rel=1e-4, abs=1e-5)


def test_gate_loop_matches_reference():
    rng = np.random.default_rng(2)
    # loud burst, then near-silence, so the gate opens, holds and closes
    x = np.concatenate([rng.standard_normal(3000) * 0.3,
                        rng.standard_normal(6000) * 0.001]).astype(np.float32)
    args = (0.1, 0.05, 0.01, 0.001, 0.0005, 0.02, 1 / 48000.0,
            0.0, 0.0, 0.0, False)
    got = effects._gate_loop(x, *args)
    ref = _py(effects._gate_loop)(x, *args)
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-4, atol=1e-5)
    assert got[4] == ref[4]
    assert got[1:4] == pytest.approx(ref[1:4], rel=1e-4, abs=1e-5)
