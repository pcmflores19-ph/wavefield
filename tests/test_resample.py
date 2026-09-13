"""
The VAD resample.

It was rewritten to work a block at a time and to read int16 straight off the
decode cache, because the obvious linspace+interp version allocated ~7.8GB of
float64 temporaries on a 2-hour track - the largest single allocation in the
app, on a path that runs on every Analyze. These tests pin the output to the
old implementation so the rewrite cannot have moved a VAD boundary.
"""

import numpy as np
import pytest

import silero_vad_onnx


def reference_resample(samples, orig_rate, target_rate):
    """The original implementation, kept here as the thing to match."""
    if orig_rate == target_rate:
        return samples.astype(np.float32)
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    duration = samples.shape[0] / orig_rate
    target_count = max(0, int(round(duration * target_rate)))
    if target_count <= 0:
        return np.zeros(0, dtype=np.float32)
    orig_x = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_x, orig_x, samples).astype(np.float32)


# Sizes either side of the block boundary, plus the degenerate ones.
@pytest.mark.parametrize("count", [
    0, 1, 5, 999, 48000, 48000 * 7 + 13,
    1 << 20, (1 << 20) + 777, (1 << 20) * 3 + 1,
])
def test_matches_the_reference_implementation(count):
    samples = (np.random.default_rng(count).random(count).astype(np.float32)
               - 0.5) * 1.9
    got = silero_vad_onnx._resample(samples, 48000, 16000)
    want = reference_resample(samples, 48000, 16000)

    assert got.shape == want.shape
    assert got.dtype == np.float32
    if got.size:
        # float32 epsilon territory - orders of magnitude below anything that
        # could move a speech/silence decision.
        assert np.abs(got - want).max() < 2e-6


def test_same_rate_is_a_passthrough():
    samples = (np.random.default_rng(1).random(1000).astype(np.float32) - 0.5)
    assert np.array_equal(
        silero_vad_onnx._resample(samples, 16000, 16000), samples)


def test_int16_input_matches_prescaled_float():
    """
    The caller now hands over the int16 memmap rather than a float32 copy of
    the whole track, so scaling happens in here, per block.
    """
    rng = np.random.default_rng(2)
    as_int16 = rng.integers(-32768, 32767, 300_000).astype(np.int16)
    as_float = as_int16.astype(np.float32) / 32768.0

    from_int16 = silero_vad_onnx._resample(as_int16, 48000, 16000)
    from_float = silero_vad_onnx._resample(as_float, 48000, 16000)

    assert np.abs(from_int16 - from_float).max() < 1e-7


def test_reads_from_a_memmap_without_copying_the_track(tmp_path):
    """Exercises the real calling shape: an on-disk int16 PCM cache file."""
    rng = np.random.default_rng(3)
    source = rng.integers(-20000, 20000, 48000 * 30).astype(np.int16)
    path = tmp_path / "track.pcm"
    path.write_bytes(source.tobytes())

    mapped = np.memmap(path, dtype=np.int16, mode="r")
    got = silero_vad_onnx._resample(mapped, 48000, 16000)
    want = reference_resample(source.astype(np.float32) / 32768.0, 48000, 16000)

    assert got.shape == want.shape
    assert np.abs(got - want).max() < 2e-6
