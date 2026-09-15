"""
The safety limiter (player.py's live monitoring, app.py's video-export mix,
audio_export.py's WAV export) used to trigger on plain sample peak
(np.abs(x).max()) with a 0 dBFS ceiling. That let audio through that never
looked like it clipped in this app, but DID clip once video_export.py's
final mux re-encoded it to AAC - a lossy codec can reconstruct material
parked at exactly 0 dBFS with 1-3 dB of inter-sample ("true peak") overshoot
after decode, which is what DaVinci Resolve then played back as audible
clipping. effects.true_peak() and the LIMITER_CEILING_DB constant fix this;
these tests pin down the property that actually matters (catches overshoot
a plain sample-peak read misses) rather than just checking the function runs.
"""

import numpy as np
import pytest

import effects


def test_true_peak_catches_inter_sample_overshoot_sample_peak_misses():
    # A sine's continuous peak equals its amplitude regardless of phase, but
    # a DISCRETE sample only reaches that peak if one happens to land
    # exactly on it. Pick a frequency/phase where none does: every sample
    # reads strictly under 1.0, while the real (and a correctly-reconstructed)
    # waveform still reaches 1.0 between them - genuine inter-sample
    # overshoot, not a synthetic edge case. This is also the reason a plain
    # LINEAR interpolation cannot work here: it's monotonic between any two
    # points, so it can never read higher than its neighboring samples and
    # therefore can never detect this - true_peak has to reconstruct a
    # bandlimited waveform (FFT zero-padding), not just interpolate one.
    sr = 48000
    n = 2048
    f = 0.3 * sr / 2  # well under Nyquist
    t = np.arange(n) / sr
    samples = np.sin(2 * np.pi * f * t + 0.37).astype(np.float64)
    sample_peak = float(np.abs(samples).max())
    assert sample_peak < 1.0

    peak = effects.true_peak(samples)
    assert peak > sample_peak
    assert peak == pytest.approx(1.0, abs=0.1)  # recovers close to the true amplitude


def test_true_peak_close_to_sample_peak_on_a_slow_signal():
    # A low-frequency sine has no fast inter-sample swing, so true peak
    # should stay close to the plain sample peak - not the multi-dB
    # overshoot the test above deliberately constructs. A generous but
    # bounded tolerance, since the FFT reconstruction treats this block as
    # one period and can read up to ~1 dB high right at its edges (see
    # true_peak's docstring) - always on the safe/conservative side.
    t = np.linspace(0, 1, 48000, endpoint=False)
    samples = (0.8 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sample_peak = float(np.abs(samples).max())

    peak = effects.true_peak(samples)
    assert sample_peak <= peak <= sample_peak * 1.2


def test_true_peak_empty_array_returns_zero():
    assert effects.true_peak(np.array([], dtype=np.float32)) == 0.0


def test_true_peak_single_sample_returns_its_own_magnitude():
    assert effects.true_peak(np.array([0.5], dtype=np.float32)) == pytest.approx(0.5)


def test_limiter_ceiling_is_below_0_dbfs():
    # Guards against an accidental revert to 0.0, which is exactly the bug
    # this constant fixes - see module docstring.
    assert effects.LIMITER_CEILING_DB < 0.0


def test_true_peak_1024_block_is_fast_enough_for_the_realtime_callback():
    # player.py's playback callback runs every ~21-23ms (blocksize=1024) and
    # calls true_peak once per active track plus once for the master mix -
    # this must stay a small fraction of that budget, not an assumption.
    import time

    rng = np.random.RandomState(0)
    block = (rng.randn(1024).astype(np.float32) * 0.5)

    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        effects.true_peak(block)
    elapsed = time.perf_counter() - start

    per_call_ms = (elapsed / iterations) * 1000.0
    assert per_call_ms < 1.0
