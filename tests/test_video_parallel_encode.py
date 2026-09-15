"""
Video segments now encode through a small bounded worker pool instead of one
ffmpeg process at a time (2026-09-14) - see video_export._encode_segments'
docstring for why segments (unlike an audio track's continuous VST stream)
are safe to parallelize.

These are unit tests against a faked `_run_encode`, not real ffmpeg - they
exist to pin down the concurrency/cancellation/retry/progress-aggregation
logic in isolation, deterministically and fast. Real encode correctness
(frame-accurate cuts, A/V sync, cleanup) is already covered end-to-end by
tests/test_video_export_integration.py, which spawns real ffmpeg through the
same `render()` entry point and therefore already exercises this pool.
"""

import pytest

import video_export


def test_cpu_encode_workers_scales_with_cores_and_caps_at_three(monkeypatch):
    monkeypatch.setattr(video_export, "_total_ram_bytes",
                        lambda: 16 * (1024 ** 3))
    monkeypatch.setattr(video_export.os, "cpu_count", lambda: 8)
    assert video_export._cpu_encode_workers() == 3


def test_cpu_encode_workers_leaves_one_core_free(monkeypatch):
    monkeypatch.setattr(video_export, "_total_ram_bytes",
                        lambda: 16 * (1024 ** 3))
    monkeypatch.setattr(video_export.os, "cpu_count", lambda: 2)
    assert video_export._cpu_encode_workers() == 1


def test_cpu_encode_workers_refuses_parallelism_on_low_ram(monkeypatch):
    monkeypatch.setattr(video_export, "_total_ram_bytes",
                        lambda: 4 * (1024 ** 3))
    monkeypatch.setattr(video_export.os, "cpu_count", lambda: 16)
    assert video_export._cpu_encode_workers() == 1


def test_cpu_encode_workers_falls_back_to_one_if_cores_unknown(monkeypatch):
    # os.cpu_count() can return None; the fallback core count (2) still
    # leaves only 1 worker once one core is reserved for the UI/OS.
    monkeypatch.setattr(video_export, "_total_ram_bytes",
                        lambda: 16 * (1024 ** 3))
    monkeypatch.setattr(video_export.os, "cpu_count", lambda: None)
    assert video_export._cpu_encode_workers() == 1


def test_gpu_encode_workers_is_conservative():
    assert video_export._gpu_encode_workers() == 2


def test_progress_aggregator_combines_multiple_concurrent_clips():
    seen = []
    agg = video_export._ProgressAggregator(
        10.0, lambda frac, msg: seen.append(frac), "Encoding video...")
    agg.track("a")(3.0)
    agg.track("b")(2.0)
    assert seen[-1] == pytest.approx(0.5)


def test_progress_aggregator_finish_pins_to_real_duration():
    seen = []
    agg = video_export._ProgressAggregator(
        10.0, lambda frac, msg: seen.append(frac), "Encoding video...")
    agg.track("a")(1.0)          # ffmpeg's last time= line understated it
    agg.finish("a", 5.0)         # pinned to the clip's real duration
    agg.track("b")(0.0)
    assert seen[-1] == pytest.approx(0.5)


def test_progress_aggregator_does_nothing_without_a_progress_callback():
    agg = video_export._ProgressAggregator(10.0, None, "Encoding video...")
    agg.track("a")(3.0)          # must not raise with progress=None
    agg.finish("a", 5.0)


def _spec(clip_id, duration=1.0):
    return (clip_id, f"clip{clip_id}.mp4", [], "vf", duration)


def test_encode_segments_runs_every_clip(monkeypatch):
    processed = []

    def fake_run_encode(cmd, out_path, on_time, should_cancel):
        processed.append(out_path)
        on_time(1.0)

    monkeypatch.setattr(video_export, "_run_encode", fake_run_encode)
    monkeypatch.setattr(video_export, "_cpu_encode_workers", lambda: 3)
    clip_specs = [_spec(i) for i in range(5)]
    agg = video_export._ProgressAggregator(5.0, None, "Encoding video...")

    video_export._encode_segments(clip_specs, ["-c:v", "libopenh264"],
                                  False, agg, None)

    assert sorted(processed) == [f"clip{i}.mp4" for i in range(5)]


def test_encode_segments_cancels_before_starting_when_already_cancelled(
        monkeypatch):
    def fake_run_encode(cmd, out_path, on_time, should_cancel):
        raise AssertionError("must not run once already cancelled")

    monkeypatch.setattr(video_export, "_run_encode", fake_run_encode)
    clip_specs = [_spec(0), _spec(1)]
    agg = video_export._ProgressAggregator(2.0, None, "Encoding video...")

    with pytest.raises(video_export._Cancelled):
        video_export._encode_segments(clip_specs, ["-c:v", "libopenh264"],
                                      False, agg, lambda: True)


def test_encode_segments_cpu_failure_raises_without_retry(monkeypatch):
    calls = {"count": 0}

    def fake_run_encode(cmd, out_path, on_time, should_cancel):
        calls["count"] += 1
        raise RuntimeError("simulated real encode failure")

    monkeypatch.setattr(video_export, "_run_encode", fake_run_encode)
    monkeypatch.setattr(video_export, "_cpu_encode_workers", lambda: 3)
    clip_specs = [_spec(0)]
    agg = video_export._ProgressAggregator(1.0, None, "Encoding video...")

    with pytest.raises(RuntimeError):
        video_export._encode_segments(clip_specs, ["-c:v", "libopenh264"],
                                      False, agg, None)
    # No GPU concurrency retry applies to the CPU path - one attempt only.
    assert calls["count"] == 1


def test_encode_segments_gpu_retries_sequentially_after_concurrent_failure(
        monkeypatch):
    """
    Simulates a GPU that fails at concurrency >1 (e.g. a single-session
    NVENC) but works fine one encode at a time - _encode_segments should
    retry the whole batch at concurrency 1 rather than failing the export.
    """
    clip_specs = [_spec(0), _spec(1), _spec(2)]
    calls = {"count": 0}

    def fake_run_encode(cmd, out_path, on_time, should_cancel):
        calls["count"] += 1
        if calls["count"] <= len(clip_specs):
            raise RuntimeError("simulated concurrent NVENC session failure")
        on_time(1.0)

    monkeypatch.setattr(video_export, "_run_encode", fake_run_encode)
    monkeypatch.setattr(video_export, "_gpu_encode_workers", lambda: 2)
    agg = video_export._ProgressAggregator(3.0, None, "Encoding video...")

    video_export._encode_segments(clip_specs, ["-c:v", "h264_nvenc"],
                                  True, agg, None)

    assert calls["count"] == 2 * len(clip_specs)


def test_encode_segments_gpu_raises_if_retry_also_fails(monkeypatch):
    def fake_run_encode(cmd, out_path, on_time, should_cancel):
        raise RuntimeError("genuine nvenc failure, not just concurrency")

    monkeypatch.setattr(video_export, "_run_encode", fake_run_encode)
    monkeypatch.setattr(video_export, "_gpu_encode_workers", lambda: 2)
    clip_specs = [_spec(0), _spec(1)]
    agg = video_export._ProgressAggregator(2.0, None, "Encoding video...")

    with pytest.raises(RuntimeError):
        video_export._encode_segments(clip_specs, ["-c:v", "h264_nvenc"],
                                      True, agg, None)
