"""
Per-clip conform (_conform_vf) and seek math (_two_stage_seek) for the
per-segment-encode + concat-demuxer video export pipeline (2026-09-13,
replacing the single shared filter_complex graph in _filter_graph).

Every segment used to run a full scale+pad letterbox chain unconditionally,
even when its source already matched the target output size exactly - a
measured contributor to a slow multi-camera export. _conform_vf skips
scale/pad when the clip's own source already matches; a mismatched source
keeps the original chain unchanged.

_conform_vf also skips the fps filter when the source's frame rate already
matches - not just an optimization: an integration test caught the fps
filter shifting a freshly-seeked segment's selected frame back by one
(2026-09-13), so skipping it when genuinely redundant is a correctness fix.
"""

import video_export


def test_conform_vf_skips_scale_and_pad_when_matching():
    vf = video_export._conform_vf(1920, 1080, "30.000000",
                                   src_width=1920, src_height=1080)

    assert "scale=" not in vf
    assert "pad=" not in vf
    assert "setsar=1" in vf
    assert "format=yuv420p" in vf
    assert "setpts=PTS-STARTPTS" in vf


def test_conform_vf_keeps_scale_and_pad_when_mismatched():
    vf = video_export._conform_vf(1920, 1080, "30.000000",
                                   src_width=1280, src_height=720)

    assert "scale=1920:1080:force_original_aspect_ratio=decrease" in vf
    assert "pad=1920:1080:-1:-1:color=black" in vf


def test_conform_vf_no_src_dims_keeps_scale_and_pad():
    """Callers that don't know the source's dimensions get the safe chain."""
    vf = video_export._conform_vf(1920, 1080, "30.000000")

    assert "scale=1920:1080:force_original_aspect_ratio=decrease" in vf
    assert "pad=1920:1080:-1:-1:color=black" in vf


def test_conform_vf_skips_fps_when_matching():
    vf = video_export._conform_vf(1920, 1080, "30.000000",
                                   fps_matches=True)

    assert "fps=" not in vf
    assert "setpts=PTS-STARTPTS" in vf


def test_conform_vf_keeps_fps_when_not_matching():
    """Default (fps_matches=False) - callers that don't know must keep it."""
    vf = video_export._conform_vf(1920, 1080, "30.000000")

    assert "fps=30.000000" in vf


def test_conform_vf_skips_everything_when_fully_matching():
    vf = video_export._conform_vf(1920, 1080, "30.000000",
                                   src_width=1920, src_height=1080,
                                   fps_matches=True)

    assert vf == "setpts=PTS-STARTPTS,setsar=1,format=yuv420p"


def test_two_stage_seek_normal_case():
    fast_seek, residual = video_export._two_stage_seek(100.0)

    assert fast_seek == 100.0 - video_export.VIDEO_SEEK_MARGIN
    assert residual == video_export.VIDEO_SEEK_MARGIN
    assert fast_seek + residual == 100.0


def test_two_stage_seek_clamped_near_start_of_file():
    """A segment starting before MARGIN must not seek to a negative time."""
    fast_seek, residual = video_export._two_stage_seek(2.0)

    assert fast_seek == 0.0
    assert residual == 2.0


def test_two_stage_seek_at_zero():
    fast_seek, residual = video_export._two_stage_seek(0.0)

    assert fast_seek == 0.0
    assert residual == 0.0


def test_bookend_input_black_reports_target_dims_and_no_fps_for_skip():
    """
    A black bookend is generated at exactly width x height x fps, so it
    should report matching dims and a None fps sentinel (meaning "always
    matches") back to the caller, rather than leaving anything ambiguous.
    """
    args, kind, width, height, fps = video_export._bookend_input(
        None, 5.0, 1920, 1080, "30.000000")

    assert kind == "black"
    assert width == 1920
    assert height == 1080
    assert fps is None
    assert "-f" in args and "lavfi" in args


def test_bookend_input_zero_seconds_is_skipped():
    args, kind, width, height, fps = video_export._bookend_input(
        None, 0.0, 1920, 1080, "30.000000")

    assert args == []
    assert kind is None
    assert width is None
    assert height is None
    assert fps is None


def test_bookend_input_video_reports_probed_dims_and_fps(monkeypatch):
    from fractions import Fraction
    from types import SimpleNamespace

    monkeypatch.setattr(video_export, "probe",
                        lambda path: SimpleNamespace(has_video=True,
                                                      width=1280, height=720,
                                                      fps=Fraction(24, 1)))

    args, kind, width, height, fps = video_export._bookend_input(
        "intro.mp4", 5.0, 1920, 1080, "30.000000")

    assert kind == "video"
    assert width == 1280
    assert height == 720
    assert fps == Fraction(24, 1)
    assert "-i" in args and "intro.mp4" in args
