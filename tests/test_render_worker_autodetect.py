"""
render_tracks' default worker count is picked from the machine's total RAM
(2026-09-13) rather than a flat constant - 2 concurrent tracks measured
~11GB peak on a 16GB+ machine, too tight a margin on anything smaller. See
audio_export._auto_render_workers's docstring for the threshold reasoning.

These monkeypatch audio_export._total_ram_bytes directly rather than
depending on the real test machine's RAM, so they're deterministic
regardless of what machine runs them.
"""

import audio_export


def test_auto_render_workers_picks_two_at_or_above_threshold(monkeypatch):
    monkeypatch.setattr(audio_export, "_total_ram_bytes",
                        lambda: audio_export._RENDER_WORKERS_RAM_THRESHOLD_BYTES)
    assert audio_export._auto_render_workers() == 2


def test_auto_render_workers_picks_two_comfortably_above_threshold(monkeypatch):
    monkeypatch.setattr(
        audio_export, "_total_ram_bytes",
        lambda: audio_export._RENDER_WORKERS_RAM_THRESHOLD_BYTES * 2)
    assert audio_export._auto_render_workers() == 2


def test_auto_render_workers_picks_one_below_threshold(monkeypatch):
    monkeypatch.setattr(
        audio_export, "_total_ram_bytes",
        lambda: audio_export._RENDER_WORKERS_RAM_THRESHOLD_BYTES - 1)
    assert audio_export._auto_render_workers() == 1


def test_auto_render_workers_picks_one_on_a_low_ram_machine(monkeypatch):
    monkeypatch.setattr(audio_export, "_total_ram_bytes",
                        lambda: 8 * (1024 ** 3))
    assert audio_export._auto_render_workers() == 1


def test_auto_render_workers_picks_three_at_the_32gb_tier(monkeypatch):
    monkeypatch.setattr(audio_export, "_total_ram_bytes",
                        lambda: 32 * (1024 ** 3))
    assert audio_export._auto_render_workers() == 3


def test_auto_render_workers_picks_two_just_below_the_32gb_tier(monkeypatch):
    monkeypatch.setattr(audio_export, "_total_ram_bytes",
                        lambda: 32 * (1024 ** 3) - 1)
    assert audio_export._auto_render_workers() == 2


def test_auto_render_workers_picks_four_at_the_48gb_tier(monkeypatch):
    monkeypatch.setattr(audio_export, "_total_ram_bytes",
                        lambda: 48 * (1024 ** 3))
    assert audio_export._auto_render_workers() == 4


def test_auto_render_workers_picks_three_just_below_the_48gb_tier(monkeypatch):
    monkeypatch.setattr(audio_export, "_total_ram_bytes",
                        lambda: 48 * (1024 ** 3) - 1)
    assert audio_export._auto_render_workers() == 3


def test_auto_render_workers_falls_back_to_two_when_ram_cannot_be_determined(monkeypatch):
    """
    A query failure isn't evidence of a memory problem - falls back to the
    known-safe default rather than silently costing everyone speed for an
    unrelated failure (unusual Windows config, running under Wine, etc.).
    """
    monkeypatch.setattr(audio_export, "_total_ram_bytes", lambda: None)
    assert audio_export._auto_render_workers() == 2


def test_total_ram_bytes_returns_none_if_the_windows_api_call_raises(monkeypatch):
    import ctypes

    class ExplodingKernel32:
        def GlobalMemoryStatusEx(self, *args, **kwargs):
            raise OSError("simulated ctypes failure")

    monkeypatch.setattr(ctypes, "windll",
                        type("FakeWindll", (), {"kernel32": ExplodingKernel32()})())

    assert audio_export._total_ram_bytes() is None


def test_total_ram_bytes_returns_none_if_the_windows_api_call_fails(monkeypatch):
    """GlobalMemoryStatusEx returning falsy (failure) rather than raising."""
    import ctypes

    class FailingKernel32:
        def GlobalMemoryStatusEx(self, *args, **kwargs):
            return 0

    monkeypatch.setattr(ctypes, "windll",
                        type("FakeWindll", (), {"kernel32": FailingKernel32()})())

    assert audio_export._total_ram_bytes() is None


def test_render_tracks_explicit_max_workers_overrides_auto_detection(monkeypatch, tmp_path):
    """
    An explicit max_workers (as tests/test_export_streaming.py already
    passes) must still work unchanged - only the *default* comes from RAM
    detection now, not the parameter itself.
    """
    import numpy as np

    monkeypatch.setattr(audio_export, "_total_ram_bytes",
                        lambda: 8 * (1024 ** 3))  # would auto-pick 1

    rng = np.random.default_rng(3)
    path_map = {}
    names = []
    for index in range(3):
        name = f"o{index}"
        samples = rng.integers(-25000, 25000, 48000 * 2, dtype=np.int16)
        path = tmp_path / f"{name}.pcm"
        path.write_bytes(samples.tobytes())
        path_map[name] = str(path)
        names.append(name)
    monkeypatch.setattr(audio_export, "decode_to_pcm", lambda n: path_map[n])

    # Just confirm this doesn't error and respects an explicit override -
    # the exact concurrency achieved isn't observable from here, only that
    # the call succeeds with a value different from what auto-detection
    # would have picked.
    mix, _rendered = audio_export.render_tracks(
        names, [(0.0, 2.0)], max_workers=2)
    assert mix.size > 0
