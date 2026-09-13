"""
Waveform peaks.

processed_peaks was changed to stream the track a chunk at a time - it used to
hold three full-length float32 arrays at once (4.15GB on a 2-hour track, to
produce 1.4MB of peaks) on a path that runs once per track on every chain
edit. The drawn picture must not change, so these pin the streamed result
against reducing the whole track in one go.
"""

import numpy as np
import pytest

import player
import waveform


def reference_peaks(samples, buckets):
    """The original all-at-once reduction, kept as the thing to match."""
    if samples.size == 0 or buckets <= 0:
        return np.zeros(max(buckets, 1), dtype=np.float32)
    per_bucket = int(np.ceil(samples.size / buckets))
    padded = np.zeros(per_bucket * buckets, dtype=np.float32)
    padded[: samples.size] = np.abs(samples)
    return padded.reshape(buckets, per_bucket).max(axis=1)


@pytest.fixture
def pcm_track(tmp_path, monkeypatch):
    """Writes an int16 PCM file and points decode_to_pcm at it."""
    def make(seconds, seed=0):
        count = int(48000 * seconds)
        # int16 straight out, or the int64 default would dwarf what's under test.
        samples = np.random.default_rng(seed).integers(
            -30000, 30000, count, dtype=np.int16)
        path = tmp_path / f"track_{seed}_{count}.pcm"
        path.write_bytes(samples.tobytes())
        monkeypatch.setattr(player, "decode_to_pcm", lambda _p: str(path))
        return samples
    return make


# Durations that land on and off a chunk boundary, and one shorter than a
# single bucket, so the zero-padded tail is covered too.
@pytest.mark.parametrize("seconds", [0.3, 1.0, 7.3, 63.0, 130.0])
def test_streamed_peaks_match_all_at_once(seconds, pcm_track):
    samples = pcm_track(seconds)

    got = waveform.processed_peaks("ignored", None, seconds)
    want = reference_peaks(samples.astype(np.float32) / 32768.0,
                           max(1, int(round(seconds * waveform.PEAKS_PER_SECOND))))

    assert got.shape == want.shape
    # Exactly equal, not merely close: chunks are cut on bucket boundaries
    # precisely so the reduction is unchanged.
    assert np.array_equal(got, want)


def test_peaks_follow_the_audio(pcm_track, tmp_path, monkeypatch):
    """A loud patch in a quiet track must show up in the right buckets."""
    count = 48000 * 4
    samples = np.zeros(count, dtype=np.int16)
    samples[48000:96000] = 20000          # second 1..2 is loud
    path = tmp_path / "spike.pcm"
    path.write_bytes(samples.tobytes())
    monkeypatch.setattr(player, "decode_to_pcm", lambda _p: str(path))

    peaks = waveform.processed_peaks("ignored", None, 4.0)

    per_second = waveform.PEAKS_PER_SECOND
    assert peaks[:per_second].max() == 0
    assert peaks[per_second:2 * per_second].min() > 0.5
    assert peaks[2 * per_second:].max() == 0


def test_runs_the_chain_through_every_chunk(pcm_track):
    """With a chain attached the peaks change, across the whole track."""
    import vst_host

    pcm_track(120.0)
    chain = vst_host.TrackChain()
    chain.add_native("limiter")

    plain = waveform.processed_peaks("ignored", None, 120.0)
    processed = waveform.processed_peaks("ignored", chain, 120.0)

    assert processed.shape == plain.shape
    # A limiter can only pull peaks down, never push them up...
    assert (processed <= plain + 1e-6).all()
    # ...and it must have been applied past the first chunk, not just to the
    # start of the track.
    assert (processed[-100:] < plain[-100:]).any()


def test_peak_memory_does_not_grow_with_track_length(pcm_track):
    """
    The point of streaming: a 10x longer track must not cost 10x the memory.

    Asserted as a ratio rather than an absolute, because the floor is set by
    the chunk size, not by how long the track is. Both lengths here span
    several chunks - a track short enough to fit in one chunk has no previous
    block alive beside the next one and so sits at half this floor, which
    would make the comparison meaningless.
    """
    import tracemalloc

    def peak_bytes_for(seconds, seed):
        pcm_track(seconds, seed=seed)
        tracemalloc.start()
        waveform.processed_peaks("ignored", None, seconds)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    shorter = peak_bytes_for(120.0, seed=1)
    longer = peak_bytes_for(600.0, seed=2)

    # 5x the audio. Before streaming this was 5x the memory.
    assert longer < shorter * 1.5, (
        f"peak grew with track length: {shorter} -> {longer} bytes")
