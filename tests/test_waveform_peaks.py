"""
Waveform peaks.

processed_peaks was changed to stream the track a chunk at a time - it used to
hold three full-length float32 arrays at once (4.15GB on a 2-hour track, to
produce 1.4MB of peaks) on a path that runs once per track on every chain
edit. The drawn picture must not change, so these pin the streamed result
against reducing the whole track in one go.

Also covers the bucket-arithmetic bug fixed here: reduce_to_peaks used to pick
a bucket COUNT from a duration in seconds, then back-derive bucket SIZE by
ceiling-dividing the sample count by that count, while the drawing code in
app.py inverted the mapping the other way (a per-second scale from the bucket
count and a duration, no ceiling). Those two formulas only agreed when a
track's sample count happened to be an exact multiple of the bucket count -
otherwise every later peak read earlier than where drawing looked for it, an
error that grew linearly with playback position. The fix (waveform.
samples_per_peak) is sample-first: a fixed sample count per bucket, with
bucket count derived from it, never the reverse - see its own docstring.
"""

import numpy as np
import pytest

import player
import waveform


def reference_peaks(samples, sample_rate, peaks_per_second):
    """The original all-at-once reduction, kept as the thing to match -
    written the same sample-first way the real reduce_to_peaks now is."""
    per_bucket = waveform.samples_per_peak(sample_rate, peaks_per_second)
    buckets = max(1, int(np.ceil(samples.size / per_bucket))) if samples.size else 1
    if samples.size == 0:
        return np.zeros(buckets, dtype=np.float32)
    padded = np.zeros(per_bucket * buckets, dtype=np.float32)
    padded[: samples.size] = np.abs(samples)
    return padded.reshape(buckets, per_bucket).max(axis=1)


@pytest.fixture
def pcm_track(tmp_path, monkeypatch):
    """
    Writes an int16 PCM file, points player.decode_to_pcm at it, and returns
    a real (but content-irrelevant) source path - waveform.py's on-disk peaks
    cache os.stat()s the ORIGINAL path before decode_to_pcm is ever consulted,
    so that path has to actually exist on disk even though its own bytes are
    never read.

    Call with `samples=` to supply an exact array (for the bucket-arithmetic
    tests below), or `seconds=`/`seed=` for a random track of that length.
    """
    counter = [0]

    def make(seconds=None, seed=0, samples=None):
        if samples is None:
            count = int(48000 * seconds)
            samples = np.random.default_rng(seed).integers(
                -30000, 30000, count, dtype=np.int16)
        counter[0] += 1
        pcm_path = tmp_path / f"track_{counter[0]}.pcm"
        pcm_path.write_bytes(samples.tobytes())
        monkeypatch.setattr(player, "decode_to_pcm", lambda _p: str(pcm_path))
        source_path = tmp_path / f"source_{counter[0]}.wav"
        source_path.write_bytes(b"\x00")
        return samples, str(source_path)
    return make


# Durations that land on and off a chunk boundary, and one shorter than a
# single bucket, so the zero-padded tail is covered too.
@pytest.mark.parametrize("seconds", [0.3, 1.0, 7.3, 63.0, 130.0])
def test_streamed_peaks_match_all_at_once(seconds, pcm_track):
    samples, source_path = pcm_track(seconds=seconds)

    got = waveform.processed_peaks(source_path, None, seconds)
    want = reference_peaks(samples.astype(np.float32) / 32768.0,
                           player.SAMPLE_RATE, waveform.PEAKS_PER_SECOND)

    assert got.shape == want.shape
    # Exactly equal, not merely close: chunks are cut on bucket boundaries
    # precisely so the reduction is unchanged.
    assert np.array_equal(got, want)


def test_peaks_follow_the_audio(pcm_track):
    """A loud patch in a quiet track must show up in the right buckets."""
    count = 48000 * 4
    samples = np.zeros(count, dtype=np.int16)
    samples[48000:96000] = 20000          # second 1..2 is loud
    _, source_path = pcm_track(samples=samples)

    peaks = waveform.processed_peaks(source_path, None, 4.0)

    per_second = waveform.PEAKS_PER_SECOND
    assert peaks[:per_second].max() == 0
    assert peaks[per_second:2 * per_second].min() > 0.5
    assert peaks[2 * per_second:].max() == 0


def test_runs_the_chain_through_every_chunk(pcm_track):
    """With a chain attached the peaks change, across the whole track."""
    import vst_host

    _, source_path = pcm_track(seconds=120.0)
    chain = vst_host.TrackChain()
    chain.add_native("limiter")

    plain = waveform.processed_peaks(source_path, None, 120.0)
    processed = waveform.processed_peaks(source_path, chain, 120.0)

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
        _, source_path = pcm_track(seconds=seconds, seed=seed)
        tracemalloc.start()
        waveform.processed_peaks(source_path, None, seconds)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    shorter = peak_bytes_for(120.0, seed=1)
    longer = peak_bytes_for(600.0, seed=2)

    # 5x the audio. Before streaming this was 5x the memory.
    assert longer < shorter * 1.5, (
        f"peak grew with track length: {shorter} -> {longer} bytes")


# ---------- bucket-arithmetic regression (the bug this fix closes) ----------

@pytest.mark.parametrize("remainder", [0, 100, 500, 959])
def test_impulse_lands_in_the_exact_bucket_at_any_remainder(remainder, pcm_track):
    """
    An impulse at a known exact sample offset must land in exactly the bucket
    that offset maps to under the fixed-size (samples_per_peak) scheme, for
    every possible remainder of (total_samples mod samples_per_peak) - not
    just the tracks that happen to divide evenly, which is exactly what let
    the old duration-first/ceil-division bug hide on some files and not
    others.
    """
    per_bucket = waveform.samples_per_peak(player.SAMPLE_RATE)
    # A track several buckets long, with a deliberately non-zero remainder.
    total_samples = per_bucket * 1500 + remainder
    seconds = total_samples / player.SAMPLE_RATE

    samples = np.zeros(total_samples, dtype=np.int16)
    impulse_sample = per_bucket * 1000 + 5      # deep into the track
    samples[impulse_sample] = 32000
    _, source_path = pcm_track(samples=samples)

    peaks = waveform.processed_peaks(source_path, None, seconds)

    expected_bucket = impulse_sample // per_bucket
    loud_buckets = np.nonzero(peaks > 0.5)[0]
    assert list(loud_buckets) == [expected_bucket], (
        f"impulse at sample {impulse_sample} (bucket {expected_bucket}) "
        f"showed up in bucket(s) {list(loud_buckets)} instead "
        f"(remainder={remainder})")


def test_no_trailing_unfilled_buckets_when_track_overshoots_bucket_count(pcm_track):
    """
    The old scheme derived bucket count from a rounded duration and could
    leave the tail of the peaks array as never-filled zeros once the
    resulting bucket size overshot past the end of the real audio. With a
    fixed per-bucket sample size and bucket count derived from the real
    sample total, every bucket up to the last must be reachable by real
    audio - fill the whole track with a constant level and check no bucket
    reads as silence.
    """
    per_bucket = waveform.samples_per_peak(player.SAMPLE_RATE)
    total_samples = per_bucket * 800 + 337
    seconds = total_samples / player.SAMPLE_RATE
    samples = np.full(total_samples, 10000, dtype=np.int16)

    _, source_path = pcm_track(samples=samples)

    peaks = waveform.processed_peaks(source_path, None, seconds)
    assert (peaks > 0).all(), "found silent (unfilled) buckets in a fully loud track"
