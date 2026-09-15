"""
sync_render._video_codec's CPU fallback (2026-09-13).

Had the identical bug as video_export.py's own CPU fallback: named
"libx264", which this bundled ffmpeg build doesn't have at all. Fixed by
routing through the same video_export.cpu_video_codec (libopenh264) - this
just confirms sync_render picks it up correctly when NVENC isn't available.
"""

import sync_render


def test_cpu_fallback_uses_openh264_not_libx264(monkeypatch):
    monkeypatch.setattr(sync_render, "has_nvenc", lambda: False)

    codec = sync_render._video_codec(1920, 1080, 30.0)

    assert "libx264" not in codec
    assert "libopenh264" in codec
    assert "-b:v" in codec


def test_gpu_path_unaffected_when_nvenc_available(monkeypatch):
    monkeypatch.setattr(sync_render, "has_nvenc", lambda: True)

    codec = sync_render._video_codec(1920, 1080, 30.0)

    assert "h264_nvenc" in codec
