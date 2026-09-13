"""
The waveform never reflects the effect chain any more (see the live-only
architecture: rendering happens only at export, and the picture is always
the real, unprocessed recording - matching Audacity/DaVinci Resolve).
`AutoCutApp._peaks_for` caches on (path, duration) only - no chain
fingerprint - so repeated calls for the same track are cheap and nothing
about editing an effect chain ever triggers a recompute, because nothing
calls this in response to a chain edit any more.
"""

import numpy as np
import pytest

import app as app_module


class _StubApp:
    """Just enough of AutoCutApp for _peaks_for to run unmodified."""

    def __init__(self):
        self._peaks_cache = {}
        self.log = lambda *a, **k: None


@pytest.fixture
def counting_processed_peaks(monkeypatch):
    calls = []

    def fake(path, chain, duration, log=None):
        calls.append((path, chain, duration))
        return np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(app_module, "processed_peaks", fake)
    return calls


def test_same_track_is_not_reprocessed(counting_processed_peaks):
    stub = _StubApp()

    app_module.AutoCutApp._peaks_for(stub, 0, "track.wav", 10.0)
    app_module.AutoCutApp._peaks_for(stub, 0, "track.wav", 10.0)

    assert len(counting_processed_peaks) == 1


def test_a_different_track_or_duration_does_reprocess(counting_processed_peaks):
    stub = _StubApp()

    app_module.AutoCutApp._peaks_for(stub, 0, "track.wav", 10.0)
    app_module.AutoCutApp._peaks_for(stub, 0, "other.wav", 10.0)
    app_module.AutoCutApp._peaks_for(stub, 0, "other.wav", 20.0)

    assert len(counting_processed_peaks) == 3


def test_never_passes_a_chain(counting_processed_peaks):
    """The whole point of the live-only architecture: the waveform is
    always the real recording, never run through the effect chain."""
    stub = _StubApp()

    app_module.AutoCutApp._peaks_for(stub, 0, "track.wav", 10.0)

    assert counting_processed_peaks[0][1] is None
