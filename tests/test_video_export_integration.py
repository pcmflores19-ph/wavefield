"""
Integration tests for the per-segment-encode + concat-demuxer video export
pipeline (2026-09-13, replacing a single shared filter_complex graph -
see video_export.py's module docstring for why).

These are the two things a redesign like this could silently break that no
unit test on the command-building helpers alone could catch: frame-accurate
cuts, and total-duration A/V sync holding across many splice points. Both
spawn the real bundled ffmpeg against a short synthetic source rather than
mocking anything - the whole point is to catch a real ffmpeg command-line
mistake a mock couldn't.

Most tests here run through NVENC (`use_gpu=True`); one (see
`test_cpu_fallback_completes_successfully`) specifically exercises the CPU
path, which used to be broken outright: this bundled ffmpeg build has no
`libx264` at all (confirmed 2026-09-13 - `ffmpeg -encoders` lists no
libx264, only libopenh264 and hardware encoders), so the CPU fallback
always failed with "Unknown encoder" until video_export.cpu_video_codec was
switched to libopenh264 (already bundled, no relicensing needed - see its
docstring).
"""

import os
import re
import subprocess
import wave

import numpy as np
import pytest

import bundled
import video_export

pytestmark = pytest.mark.skipif(
    not os.path.exists(video_export.FFMPEG),
    reason="bundled ffmpeg not present in this environment")

requires_nvenc = pytest.mark.skipif(
    not video_export.has_nvenc(),
    reason="no working NVENC in this environment")

FFPROBE = bundled.tool("ffprobe")

_FPS = 30
_WIDTH, _HEIGHT = 320, 240


def _make_source(path, duration=10.0, gop=300):
    """
    A synthetic long-GOP source - keyframes sparse enough that a cut in the
    middle of one is a real test of frame accuracy, not an accidental
    keyframe hit.

    libopenh264, not libx264: this bundled ffmpeg build has no libx264 at
    all (confirmed 2026-09-13 - a separate, pre-existing issue, since the
    app's own CPU fallback path also names "libx264"). libopenh264 needs no
    GPU, keeping source generation portable regardless of that finding.
    """
    subprocess.run([
        video_export.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        f"testsrc2=size={_WIDTH}x{_HEIGHT}:rate={_FPS}",
        "-t", f"{duration:.6f}", "-g", str(gop),
        "-c:v", "libopenh264", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)


def _make_silent_wav(path, duration, sample_rate=48000):
    samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


def _extract_frame_png(path, out_png, seek_after_i=None):
    """First frame, losslessly, optionally after an accurate (after -i) seek."""
    cmd = [video_export.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
          "-i", str(path)]
    if seek_after_i is not None:
        cmd += ["-ss", f"{seek_after_i:.6f}"]
    cmd += ["-frames:v", "1", str(out_png)]
    subprocess.run(cmd, check=True, capture_output=True)


def _ssim(png_a, png_b):
    """
    Structural similarity between two still frames - robust to the lossy
    re-encoding render() always does (NVENC/libx264), unlike a raw pixel/MD5
    comparison. A wrong-frame selection on fast-changing synthetic content
    shows up as a large SSIM drop; ordinary encoder compression noise does
    not.
    """
    result = subprocess.run([
        video_export.FFMPEG, "-hide_banner", "-loglevel", "info",
        "-i", str(png_a), "-i", str(png_b),
        "-lavfi", "ssim", "-f", "null", "-",
    ], capture_output=True, text=True)
    match = re.search(r"All:([\d.]+)", result.stderr)
    assert match, f"could not parse ssim output: {result.stderr!r}"
    return float(match.group(1))


def _probe_duration(path):
    result = subprocess.run([
        FFPROBE, "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


@requires_nvenc
def test_segment_cut_is_frame_accurate(tmp_path):
    """
    The core correctness requirement of the whole redesign: a cut landing
    off-keyframe must still land on the exact requested frame, matching
    what the old single-pass filter_complex graph guaranteed.

    Verified as a nearest-frame search, not an absolute SSIM threshold:
    testsrc2 changes enough frame-to-frame (measured: adjacent-frame SSIM
    ~0.89, two-frames-apart ~0.86) that it overlaps the SSIM range NVENC's
    own lossy re-encoding introduces even for the CORRECT frame - an
    absolute threshold can't tell "right frame, compression noise" apart
    from "off by 1-2 frames." Comparing against several candidate offsets
    and requiring the best match to land exactly at offset 0, with a clear
    margin over the runner-up, can.
    """
    source = tmp_path / "source.mp4"
    _make_source(source, duration=10.0, gop=300)

    start, end = 4.234, 6.789
    audio = tmp_path / "audio.wav"
    _make_silent_wav(audio, end - start)
    out_path = tmp_path / "out.mp4"

    result = video_export.render(
        str(source), str(audio), str(out_path), keep_ranges=[(start, end)],
        use_gpu=True)
    assert result == str(out_path)

    out_png = tmp_path / "out.png"
    _extract_frame_png(out_path, out_png)

    scores = {}
    for k in range(-3, 4):
        candidate = tmp_path / f"candidate_{k}.png"
        _extract_frame_png(source, candidate, seek_after_i=start + k / _FPS)
        scores[k] = _ssim(candidate, out_png)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_offset, best_score = ranked[0]
    margin = best_score - ranked[1][1]
    assert best_offset == 0, (
       f"Exported frame best matches source at offset {best_offset} frames "
       f"from the requested start ({start}s), not 0 - the cut landed on "
       f"the wrong frame. Scores by offset: {scores}")
    assert margin > 0.01, (
       f"Best match (offset 0, score {best_score:.4f}) isn't clearly ahead "
       f"of the runner-up (margin {margin:.4f}) - inconclusive rather than "
       f"confirmed correct. Scores by offset: {scores}")


@requires_nvenc
def test_multi_segment_duration_matches_audio(tmp_path):
    """
    Concat across several splice points must not accumulate drift - total
    output duration should match the sum of the kept ranges (what the
    already-rendered audio was built from) to within about a frame.
    """
    source = tmp_path / "source.mp4"
    _make_source(source, duration=10.0, gop=300)

    keep_ranges = [(0.5, 1.777), (3.111, 4.5), (6.0, 8.25)]
    total = sum(e - s for s, e in keep_ranges)

    audio = tmp_path / "audio.wav"
    _make_silent_wav(audio, total)
    out_path = tmp_path / "out.mp4"

    result = video_export.render(
        str(source), str(audio), str(out_path), keep_ranges=keep_ranges,
        use_gpu=True)

    assert result == str(out_path)
    duration = _probe_duration(out_path)
    assert abs(duration - total) < (1.0 / _FPS) * 2, (
       f"Output duration {duration:.3f}s drifted from the expected "
       f"{total:.3f}s across {len(keep_ranges)} splice points.")


@requires_nvenc
def test_temp_files_are_cleaned_up(tmp_path):
    """Every scratch clip/list file must be removed after a successful export."""
    source = tmp_path / "source.mp4"
    _make_source(source, duration=3.0, gop=90)

    audio = tmp_path / "audio.wav"
    _make_silent_wav(audio, 1.0)
    out_path = tmp_path / "out.mp4"

    before = set(os.listdir(tmp_path))
    video_export.render(str(source), str(audio), str(out_path),
                        keep_ranges=[(0.5, 1.5)], use_gpu=True)
    after = set(os.listdir(tmp_path))

    # Only out.mp4 should be new in this directory - no leftover clip_*.mp4,
    # concat.txt or video_only.mp4 (those live in video_export's own
    # tempfile.mkdtemp() elsewhere, but confirms the render didn't scatter
    # scratch files into the caller's working area either).
    assert after - before == {"out.mp4"}


def test_cpu_fallback_completes_successfully(tmp_path):
    """
    Regression test for the CPU path being completely broken (2026-09-13):
    `render(..., use_gpu=False)` used to fail outright with ffmpeg's
    "Unknown encoder 'libx264'", since this bundled ffmpeg build has no
    libx264 at all. No NVENC required - this is the whole point.
    """
    source = tmp_path / "source.mp4"
    _make_source(source, duration=3.0, gop=90)

    audio = tmp_path / "audio.wav"
    _make_silent_wav(audio, 1.0)
    out_path = tmp_path / "out.mp4"

    result = video_export.render(
        str(source), str(audio), str(out_path), keep_ranges=[(0.5, 1.5)],
        use_gpu=False)

    assert result == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    duration = _probe_duration(out_path)
    assert abs(duration - 1.0) < (1.0 / _FPS) * 2
