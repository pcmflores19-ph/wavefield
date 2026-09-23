"""
Waveform peaks for drawing.

The waveform is the real recording with the speaker's own VST chain on top,
never the denoised copy the analysis works from - see processed_peaks.
"""

import hashlib
import os

import numpy as np

import settings

# Peaks are extracted once at this resolution and re-bucketed in the UI when
# zooming, so zooming never needs another decode. 50/s = 20ms per peak, fine
# down to word-level zoom, and only ~175k floats for a 58-minute episode.
PEAKS_PER_SECOND = 50

# How much audio is read, processed and reduced to peaks at a time. Matches
# vst_host.CHUNK_SECONDS, which is the size that module already feeds a plugin
# when it cannot take a whole track. ~1.4M samples, so ~6MB of float32 per
# pass instead of the whole episode.
CHUNK_SECONDS = 30.0


def _bucket_maxima(block, buckets, per_bucket):
    """
    Peak magnitude of each `per_bucket`-sample group in `block`.

    `block` is consumed (made absolute in place) - callers here own it. The
    tail is zero-padded to fill the last bucket, which is what makes a
    streamed pass agree exactly with reducing the whole track at once.
    """
    need = buckets * per_bucket
    if block.size < need:
        block = np.concatenate(
            [block, np.zeros(need - block.size, dtype=np.float32)])
    np.abs(block, out=block)
    return block.reshape(buckets, per_bucket).max(axis=1)


def reduce_to_peaks(samples, total, duration_seconds, sample_rate,
                    peaks_per_second=PEAKS_PER_SECOND, offline=None, log=None):
    """
    The chunked peak-reduction loop shared by processed_peaks (in-process,
    no plugins or a live chain via chain.snapshot()) and
    vst_host._isolated_peaks_worker (a detached chain rebuilt in a fresh
    child process - see that function for why plugins can't be loaded here
    on the redraw thread). `samples` is int16 (a memmap straight off the
    decode cache, or an equivalent array); `offline`, if given, is a
    TrackChain-like object with the plugins already loaded, run once per
    chunk exactly as processed_peaks always has.
    """
    buckets = max(1, int(round(duration_seconds * peaks_per_second)))
    if total == 0:
        return np.zeros(buckets, dtype=np.float32)

    per_bucket = int(np.ceil(total / buckets))
    # Whole buckets per pass, so every bucket's peak still sees all of its
    # samples and the result matches an all-at-once reduction exactly.
    step = max(1, int(round(CHUNK_SECONDS * sample_rate / per_bucket)))

    peaks = np.zeros(buckets, dtype=np.float32)
    for first in range(0, buckets, step):
        last = min(first + step, buckets)
        start = first * per_bucket
        if start >= total:
            break
        block = np.asarray(samples[start:min(last * per_bucket, total)],
                           dtype=np.float32)
        block *= 1.0 / 32768.0

        if offline is not None:
            # reset only on the first chunk, so plugin state carries across
            # the joins - the same way vst_host chunks a plugin that cannot
            # take a whole track. log only there too, or a chatty plugin
            # would say the same thing once per chunk.
            processed = offline.process(block, sample_rate, reset=(first == 0),
                                        log=log if first == 0 else None)
            if processed.size == block.size:
                block = processed

        peaks[first:last] = _bucket_maxima(block, last - first, per_bucket)

    return peaks


def _peaks_cache_path(path, duration_seconds, peaks_per_second):
    stat = os.stat(path)
    key = hashlib.sha1(
        f"{path}|{stat.st_size}|{stat.st_mtime}|{duration_seconds}|"
        f"{peaks_per_second}|peaks".encode("utf-8")
    ).hexdigest()
    directory = settings.cache_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, key + ".peaks.f32")


def processed_peaks(path, chain, duration_seconds, log=None,
                    peaks_per_second=PEAKS_PER_SECOND):
    """
    Peaks for one speaker, for the drawn waveform.

    Deliberately the RAW file, not the denoised/levelled copy the analysis
    uses. The waveform has to be the audio - a picture normalized to -14 LUFS
    can never show clipping, and a picture already denoised makes the user's
    own denoiser look like it does nothing. What the analysis does to decide
    the cuts is its own business; see voice_activity.clean_for_analysis.

    `chain`, if given, runs the audio through it first (via chain.snapshot(),
    a detached copy - the live chain belongs to the audio callback) before
    reducing to peaks. app.py's only caller always passes None: the drawn
    waveform never reflects an edited/live effect chain, matching Audacity/
    DaVinci Resolve - only the real recording is ever pictured. The
    capability is kept here (used by streamed callers that do want a
    processed picture, and covered by tests/test_waveform_peaks.py) rather
    than removed outright.

    Streamed a chunk at a time - reducing a 2-hour track at once needed three
    full-length float32 arrays alive together (4.15GB, to produce 1.4MB of
    peaks). Chunk boundaries are invisible here: the output is one value per
    20ms, and the same audio is already monitored through 21ms blocks
    (player.py's stream blocksize).

    Disk-cached only for the real `chain=None` path (the only one app.py
    ever calls) - a track reopened after the app restarts skips this pass
    entirely instead of redecoding and re-reducing an hour of audio just to
    draw the same picture again. Keyed like voice_activity's caches
    (content hash of path/size/mtime), plus duration and peaks_per_second
    since both change how many buckets the same audio reduces to.
    """
    cache_path = None
    if chain is None:
        cache_path = _peaks_cache_path(path, duration_seconds, peaks_per_second)
        if os.path.exists(cache_path):
            try:
                cached = np.fromfile(cache_path, dtype=np.float32)
                if cached.size:
                    return cached
            except Exception:
                pass                        # fall through and rebuild it

    from player import SAMPLE_RATE, decode_to_pcm

    samples = np.memmap(decode_to_pcm(path), dtype=np.int16, mode="r")
    total = samples.shape[0]

    # A detached copy: the live chain belongs to the audio callback, and
    # driving the same VST from two threads at once crashes the process.
    offline = None
    if chain is not None and chain.active_slots():
        offline = chain.snapshot(log=log)

    peaks = reduce_to_peaks(samples, total, duration_seconds, SAMPLE_RATE,
                            peaks_per_second=peaks_per_second, offline=offline,
                            log=log)
    if cache_path is not None:
        try:
            # Atomic like voice_activity's caches: a concurrent reader (e.g.
            # this same speaker's cache-hit check above, on another analysis
            # thread) never sees a partially-written file.
            tmp_path = cache_path + ".part"
            peaks.astype(np.float32).tofile(tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception:
            pass                            # a missed cache write is not fatal
    return peaks
