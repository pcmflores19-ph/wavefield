"""
Finds who is speaking when, from the waveform alone.

This used to be derived from WhisperX word timestamps, which meant you had to
sit through a transcription of every track before you could see a single cut.
It also made the cuts only as good as the transcript: a misheard Taglish phrase
or a dropped word moved the edit. The waveform already knows where speech is,
and it knows immediately.

The analysis chain, per track (see `clean_for_analysis`):

  1. DENOISE with rnnoise - room tone, fan noise and laptop hum are what a
     plain energy gate mistakes for talking. Stripping them makes the gap
     between speech and silence wide and obvious.
  2. COMPRESS - a real compressor, threshold set from that track's own 80th
     percentile of frame level so it adapts to how the person happened to be
     recorded, ratio 3:1.
  3. LOUDNESS-NORMALIZE to -14 LUFS, so every recording ends up levelled the
     same way regardless of room, mic or Meet session.
  4. THRESHOLD - an adaptive gate with hysteresis over short frames.

Steps 1-3 exist ONLY to make the decision (and to draw the waveform - see
`clean_for_analysis`'s callers in app.py and waveform.py). Nothing here is
baked into the track: what comes back is a list of time ranges, or a throwaway
processed copy for display. The audio you hear, edit and export is untouched
by any of it - the user's own VST chain remains the only thing that ever
changes the sound.
"""

import hashlib
import json
import multiprocessing
import os
import tempfile

import numpy as np

import effects
import settings
from player import SAMPLE_RATE, decode_to_pcm

FRAME_SECONDS = 0.020        # 20ms frames, 10ms hop - fine enough to catch
HOP_SECONDS = 0.010          # word boundaries without chasing every glottal stop

# Where the gate sits between the measured noise floor and the measured speech
# level. 0.30 puts it nearer the floor, which is right after denoising: the
# floor is genuinely quiet, so the risk is clipping soft speech, not letting
# noise through.
THRESHOLD_FRACTION = 0.30
MIN_MARGIN_DB = 8.0          # never gate less than this above the noise floor
ABSOLUTE_FLOOR_DB = -55.0    # nothing quieter than this is ever speech

# rnnoise doesn't attenuate room tone, it removes it - the quiet parts come back
# as true digital silence, so the measured "noise floor" is the -180 dB clamp
# and floor-relative maths stops meaning anything. This keeps the gate a fixed
# distance BELOW the speech level, which stays meaningful either way.
BELOW_SPEECH_DB = 20.0

# Hysteresis: once speech has started it takes a bigger drop to end it, so
# ordinary dips inside a word don't chop it in half.
RELEASE_DB = 5.0

# Speech separated by less than this is treated as one region. Without it the
# gate flickers: a lip smack or a dip inside a word closes and reopens it, and
# a single 20ms blip in the middle of a long silence splits that silence into
# two halves that are each too short to cut. Merging first, then discarding the
# stragglers, is what stops one stray sample protecting five seconds of dead
# air - and it is far gentler than simply demanding long speech regions, which
# throws away real one-word answers ("oo", "tama").
# Each detected region is widened by this before anything else. A word does not
# start at full volume - "s", "f", "h" and a soft first syllable climb past the
# gate a moment after the sound actually began, and the tail of a word fades
# below it before the word is over. Detecting from energy alone therefore always
# lands slightly INSIDE the word at both ends, and cutting there clips the first
# or last letter. Widening first means the speech regions bracket the whole word.
ONSET_GUARD_SECONDS = 0.10

HANGOVER_SECONDS = 0.15
MIN_SPEECH_SECONDS = 0.20    # shorter than this, after merging, is not a word

# How many 20ms frames _frame_levels squares at a time - see the comment there.
# 8192 frames is ~63MB of float64 scratch, small enough to be irrelevant next
# to the track itself and large enough that the loop overhead disappears.
_RMS_CHUNK_FRAMES = 8192


def _load(path):
    """The track as mono float32, straight off the decode cache."""
    samples = np.memmap(decode_to_pcm(path), dtype=np.int16, mode="r")
    return np.asarray(samples, dtype=np.float32) / 32768.0


def _frame_levels(samples):
    """RMS per frame, in dBFS. Returns (levels_db, hop_seconds)."""
    frame = int(FRAME_SECONDS * SAMPLE_RATE)
    hop = int(HOP_SECONDS * SAMPLE_RATE)
    if samples.size < frame:
        return np.zeros(0, dtype=np.float32), HOP_SECONDS

    count = 1 + (samples.size - frame) // hop
    # A strided view: no copy, so an hour of audio costs nothing extra here.
    frames = np.lib.stride_tricks.as_strided(
        samples, shape=(count, frame),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    # Squared in blocks of frames, not all at once. The view is free, but
    # np.square(frames, dtype=np.float64) is not: frames overlap 2:1, so it
    # materialises ~2x the track as float64 - measured at 2.9GB for one
    # 67-minute track, which is most of what took the process down with an
    # access violation partway through a 3-speaker analysis (see
    # autocut_crash.log; native code faults where Python would raise
    # MemoryError). Each frame's RMS is independent of every other, so this
    # is the same arithmetic in the same order, just bounded.
    rms = np.empty(count, dtype=np.float64)
    for start in range(0, count, _RMS_CHUNK_FRAMES):
        block = frames[start:start + _RMS_CHUNK_FRAMES]
        rms[start:start + block.shape[0]] = np.sqrt(
            np.mean(np.square(block, dtype=np.float64), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9)), HOP_SECONDS


def find_denoiser(plugins=None):
    """The rnnoise VST3 path, or None if it isn't installed on this machine."""
    if plugins is None:
        import vst_host
        plugins = vst_host.discover_plugins()
    for name, path in plugins:
        if "rnnoise" in name.lower():
            return path
    return None


# How long to wait for the isolated denoise worker below before giving up and
# falling back to raw audio - matches silero_vad_onnx's own worker timeout.
_DENOISE_WORKER_TIMEOUT_SECONDS = 300.0


def _denoise_worker(input_path, output_path, count, plugin_path, sample_rate,
                    conn):
    """
    Runs in the child process: loads rnnoise fresh and runs it start to
    finish, reading/writing disk-backed memmaps rather than pickling the
    track through `conn` - see _denoise_isolated for why. Sends
    ("ok", None) or ("error", message); the parent reads output_path itself.
    """
    try:
        import vst_host
        samples = np.memmap(input_path, dtype=np.float32, mode="r",
                            shape=(count,))
        chain = vst_host.TrackChain()
        chain.add("rnnoise", plugin_path)
        result = chain.process(samples, sample_rate, reset=True)
        if result.size != count:
            raise RuntimeError("denoised audio length did not match the source")
        output = np.memmap(output_path, dtype=np.float32, mode="w+",
                           shape=(count,))
        output[:] = result
        output.flush()
        del output
        del samples
        conn.send(("ok", None))
    except Exception as exc:
        conn.send(("error", str(exc)))
    finally:
        conn.close()


def _denoise_isolated(samples, plugin_path, sample_rate):
    """
    Runs rnnoise in a fresh child process - same pattern as
    silero_vad_onnx.speech_probabilities_isolated, and for a related but
    distinct reason.

    Reproduced directly (2026-09-11): loading rnnoise via
    vst_host.TrackChain.add()'s main-thread hop, once per speaker, while ALSO
    spawning a separate child process per speaker for the isolated Silero VAD
    call, hangs the app by the 3rd speaker - confirmed by elimination: an
    otherwise-identical repro survived every time once the hop was the only
    thing removed. A fresh child process here never touches the main thread's
    plugin-hosting state at all (its own single thread IS its "main thread",
    with nothing else ever having claimed it), which is exactly the condition
    that survived in that reproduction - not a workaround, the same fix shape
    already proven for Silero.

    The array itself crosses into and back out of that process via disk-
    backed memmaps, not `Process(args=...)` or the pipe - passing the full
    track either way pickles three copies of it (parent + pickle buffer +
    child) into existence at once, per speaker, which is what actually
    crashed/froze long multi-speaker sessions.
    """
    directory = settings.cache_dir()
    os.makedirs(directory, exist_ok=True)
    fd_in, in_path = tempfile.mkstemp(dir=directory, suffix=".denoise_in.pcm")
    os.close(fd_in)
    fd_out, out_path = tempfile.mkstemp(dir=directory, suffix=".denoise_out.pcm")
    os.close(fd_out)
    try:
        count = samples.shape[0]
        memmap_in = np.memmap(in_path, dtype=np.float32, mode="w+",
                              shape=(count,))
        block = _RMS_CHUNK_FRAMES * 64
        for start in range(0, count, block):
            stop = min(start + block, count)
            memmap_in[start:stop] = samples[start:stop]
        memmap_in.flush()
        del memmap_in

        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        proc = multiprocessing.Process(
            target=_denoise_worker,
            args=(in_path, out_path, count, plugin_path, sample_rate,
                  child_conn),
            daemon=True)
        proc.start()
        child_conn.close()  # only the child should hold the writable end
        try:
            if not parent_conn.poll(_DENOISE_WORKER_TIMEOUT_SECONDS):
                proc.terminate()
                proc.join(5.0)
                raise RuntimeError("rnnoise worker timed out")
            try:
                status, payload = parent_conn.recv()
            except EOFError:
                proc.join(5.0)
                raise RuntimeError(
                    f"rnnoise worker crashed (exit code {proc.exitcode})")
        finally:
            parent_conn.close()
        proc.join(5.0)
        if status != "ok":
            raise RuntimeError(f"rnnoise failed in worker process: {payload}")

        # Owned, not a memmap: _lufs_normalize mutates its input in place,
        # and the backing file is removed in `finally` below.
        result_mm = np.memmap(out_path, dtype=np.float32, mode="r",
                              shape=(count,))
        result = np.array(result_mm, dtype=np.float32)
        del result_mm
        return result
    finally:
        for path in (in_path, out_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def denoise(samples, plugin_path, log=None):
    """
    Runs the track through rnnoise for analysis purposes only.

    Isolated in its own process, not just its own chain - see
    _denoise_isolated's docstring for why that is load-bearing, not just
    tidy: never the user's live chain either way, which belongs to the audio
    callback.
    """
    try:
        return _denoise_isolated(samples, plugin_path, SAMPLE_RATE)
    except Exception as exc:
        if log:
            log(f"  rnnoise unavailable ({exc}); detecting on the raw waveform")
        return samples


# ------------------------------------------------------- loudness (K-weighting)
#
# A hand-rolled approximation of ITU-R BS.1770's loudness meter: the same two
# cascaded filters (a high-shelf "pre-filter" standing in for the head, then a
# high-pass "RLB" filter), designed here from the standard's published analog
# parameters via the RBJ Audio EQ Cookbook's biquad formulas, rather than
# copying the spec's fixed 48kHz coefficient table - this way it is correct at
# whatever SAMPLE_RATE the app actually decodes to.
#
# This is a good-faith implementation for deciding an internal gain, not a
# certified loudness meter: the number it produces is never shown or exported,
# only used to level a throwaway analysis copy. The two-stage gating below
# (absolute, then relative) is real BS.1770 gating, not a simplification -
# skipping it would badly overstate the loudness of a track that is mostly one
# person's silence while the other speaks, which is the normal shape of a
# per-speaker podcast recording.

_SHELF_F0, _SHELF_DB, _SHELF_Q = 1681.9744509555319, 3.99984385397, 0.7071752369554193
_HPF_F0, _HPF_Q = 38.13547087613982, 0.5003270373238773


def _high_shelf_coeffs(f0, db_gain, q, sample_rate):
    a = 10.0 ** (db_gain / 40.0)
    w0 = 2.0 * np.pi * f0 / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(a)
    b0 = a * ((a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cos_w0)
    b2 = a * ((a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cos_w0)
    a2 = (a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _high_pass_coeffs(f0, q, sample_rate):
    w0 = 2.0 * np.pi * f0 / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _biquad(samples, b0, b1, b2, a1, a2):
    """Direct-form-I biquad, sample by sample - genuinely sequential, as
    effects._envelope is."""
    out = np.empty_like(samples)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(samples.size):
        x0 = samples[i]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y0
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0
    return out


try:                                    # same optional speedup as effects.py
    from numba import njit
    _biquad = njit(cache=True, fastmath=True)(_biquad)
except Exception:
    pass


def _k_weight(samples, sample_rate):
    shelf = _high_shelf_coeffs(_SHELF_F0, _SHELF_DB, _SHELF_Q, sample_rate)
    hpf = _high_pass_coeffs(_HPF_F0, _HPF_Q, sample_rate)
    stage1 = _biquad(samples.astype(np.float32), *shelf)
    return _biquad(stage1, *hpf)


# _k_weight is the expensive part of measuring loudness - a genuinely
# sequential biquad, sample by sample, with no numba here to speed it up (see
# clean_for_analysis's docstring on why one isn't added). Running it over
# literally every sample of an hour-long track cost minutes per speaker.
# Integrated loudness is already a statistic over many 400ms blocks, so past
# a certain length we K-weight a representative, evenly-spaced sample of the
# track instead of all of it - the same gating decides the result either way,
# just from fewer blocks. Short tracks are still measured in full.
_LUFS_FULL_SCAN_SECONDS = 20.0     # tracks up to this long: no sampling
_LUFS_WINDOW_SECONDS = 2.0         # length of each sampled window
_LUFS_SAMPLE_COVERAGE = 0.25       # fraction of a long track actually scanned


def _block_mean_squares(weighted, sample_rate):
    """Mean square per 400ms block on a 100ms hop, for one contiguous
    K-weighted signal."""
    block = int(0.400 * sample_rate)
    hop = int(0.100 * sample_rate)
    if weighted.size < block:
        block = weighted.size
        hop = max(1, block)
    if block <= 0:
        return np.zeros(0, dtype=np.float64)
    count = 1 + max(0, (weighted.size - block) // hop)
    blocks = np.lib.stride_tricks.as_strided(
        weighted, shape=(count, block),
        strides=(weighted.strides[0] * hop, weighted.strides[0]),
        writeable=False,
    )
    return np.mean(np.square(blocks, dtype=np.float64), axis=1)


def _sampled_windows(total_samples, sample_rate, window_seconds, coverage):
    """Evenly-spaced (start, end) sample ranges covering `coverage` of the
    track. Falls back to the whole track if it's too short to bother."""
    window = int(window_seconds * sample_rate)
    if window <= 0 or window >= total_samples:
        return [(0, total_samples)]
    n_windows = max(1, int(round((total_samples / window) * coverage)))
    if n_windows * window >= total_samples:
        return [(0, total_samples)]
    starts = sorted(set(
        int(s) for s in np.linspace(0, total_samples - window, n_windows)))
    return [(s, s + window) for s in starts]


def _measure_lufs(samples, sample_rate):
    """
    Integrated loudness of a mono track, in LUFS - ITU-R BS.1770's formula
    (-0.691 + 10*log10(mean square)) over 400ms blocks on a 100ms hop, with
    the standard's two-stage gating: blocks quieter than -70 LUFS are dropped
    outright (that is analogue noise floor, not programme), then blocks more
    than 10 LU below the loudness of what's left are dropped too, so long
    stretches of one speaker's silence don't drag the estimate down.
    """
    if samples.size == 0:
        return -70.0

    duration = samples.size / sample_rate
    if duration <= _LUFS_FULL_SCAN_SECONDS:
        mean_sq = _block_mean_squares(_k_weight(samples, sample_rate), sample_rate)
    else:
        windows = _sampled_windows(samples.size, sample_rate,
                                   _LUFS_WINDOW_SECONDS, _LUFS_SAMPLE_COVERAGE)
        parts = [_block_mean_squares(
                    _k_weight(np.ascontiguousarray(samples[start:end]), sample_rate),
                    sample_rate)
                 for start, end in windows]
        mean_sq = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    mean_sq = mean_sq[mean_sq > 0]
    if mean_sq.size == 0:
        return -70.0
    block_loudness = -0.691 + 10.0 * np.log10(mean_sq)

    absolute_gated = mean_sq[block_loudness >= -70.0]
    if absolute_gated.size == 0:
        return -70.0
    relative_threshold = -0.691 + 10.0 * np.log10(np.mean(absolute_gated)) - 10.0
    relative_gated = mean_sq[
        (block_loudness >= -70.0) & (block_loudness >= relative_threshold)]
    if relative_gated.size == 0:
        relative_gated = absolute_gated
    return float(-0.691 + 10.0 * np.log10(np.mean(relative_gated)))


def _lufs_normalize(samples, sample_rate, target_lufs=-14.0):
    """
    Gains `samples` so its integrated loudness reaches `target_lufs`.

    Scales IN PLACE: the only caller is clean_for_analysis, which hands over a
    throwaway copy it drops on return, and on an hour-long track a second
    full-length array here is 0.72GB for nothing.
    """
    current = _measure_lufs(samples, sample_rate)
    if current <= -69.0:              # nothing there worth levelling
        return samples
    gain = 10.0 ** ((target_lufs - current) / 20.0)
    # Chunked so the |x| scan does not materialise a second copy of the track.
    peak = 0.0
    for start in range(0, samples.size, _RMS_CHUNK_FRAMES * 64):
        block = samples[start:start + _RMS_CHUNK_FRAMES * 64]
        peak = max(peak, float(np.abs(block).max()))
    gain = min(gain, 0.99 / (peak or 1.0))   # never clip chasing a loud target
    samples *= np.float32(gain)
    return samples


def _percentile_threshold_db(levels_db, percentile=80.0):
    """A compressor threshold that adapts to how this track was recorded,
    rather than a fixed dB number."""
    if levels_db.size == 0:
        return -20.0
    return float(np.percentile(levels_db, percentile))


def clean_for_analysis(samples, sample_rate, plugin_path=None, log=None):
    """
    Denoise -> compress -> loudness-normalize: one clean, comparable copy of
    a track, shared by speech detection, auto-mute, and the on-screen
    waveform, so what you see matches what the app decided.

    Never touches the file or the audio the app plays and exports - see the
    module docstring.
    """
    work = samples
    if plugin_path:
        work = denoise(work, plugin_path, log=log)
    # Dead weight once `work` exists (denoise's own result, or - on the
    # fallback path - the same object `work` already references): the caller
    # must not keep its own copy of `samples` alive either, see
    # cleaned_samples_for, or this full-track array stays resident through
    # compress+lufs-normalize for nothing.
    del samples

    levels_db, _ = _frame_levels(work)
    threshold_db = _percentile_threshold_db(levels_db, 80.0)
    work = effects.compressor(work, sample_rate, threshold_db=threshold_db,
                              ratio=3.0)

    return _lufs_normalize(work, sample_rate, target_lufs=-14.0)


def _cleaned_cache_path(path):
    stat = os.stat(path)
    # "clean2" (not "clean") marks the int16 format below - so a cache file
    # written by an earlier build (float32) is never misread as this one's
    # format. It just goes unused and ages out via settings.prune_cache.
    key = hashlib.sha1(
        f"{path}|{stat.st_size}|{stat.st_mtime}|clean2|{SAMPLE_RATE}".encode("utf-8")
    ).hexdigest()
    directory = settings.cache_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, key + ".clean16.pcm")


def cleaned_samples_for(path, denoiser_path=None, log=None):
    """
    `clean_for_analysis`, cached per file (same content-hash pattern, and the
    same int16 storage, as player.decode_to_pcm - halves the disk cost of a
    float32 cache, and _lufs_normalize always leaves headroom below full
    scale so nothing clips on the way down). Speech detection, auto-mute and
    the waveform all end up asking for the same speaker's cleaned audio more
    than once in a session - reopening a project, or redrawing after an edit
    - and denoising plus compressing an hour of audio is not free.
    """
    # Every conversion below is done in place or in blocks. An hour-long track
    # is 0.72GB per float32 copy, and the casual `x.astype(float32) / 32768.0`
    # this used to do held three of them at once - the kind of arithmetic that
    # added up to the out-of-memory native crash in autocut_crash.log.
    cache_path = _cleaned_cache_path(path)
    if os.path.exists(cache_path):
        try:
            stored = np.fromfile(cache_path, dtype=np.int16)
            if stored.size:
                restored = stored.astype(np.float32)
                del stored
                restored /= 32768.0
                return restored
        except Exception:
            pass                            # fall through and rebuild it
    # Not bound to a local first: clean_for_analysis drops its own reference
    # once denoise's result exists (see its own del samples), and binding it
    # here too would keep the raw array alive for the whole denoise+compress+
    # lufs pipeline for nothing - two stack frames, one full track, held
    # twice as long as needed.
    cleaned = clean_for_analysis(_load(path), SAMPLE_RATE, denoiser_path,
                                 log=log)
    cleaned = np.asarray(cleaned, dtype=np.float32)
    try:
        quantized = np.empty(cleaned.size, dtype=np.int16)
        block = _RMS_CHUNK_FRAMES * 64
        for start in range(0, cleaned.size, block):
            piece = cleaned[start:start + block]
            quantized[start:start + piece.size] = np.clip(
                piece * 32767.0, -32768, 32767).astype(np.int16)
        # Atomic like player.decode_to_pcm's cache write: writing straight to
        # cache_path let a concurrent reader (e.g. this same speaker's cache
        # hit path above, on another thread) np.fromfile a partially-written
        # file. os.replace is a single filesystem operation - a reader either
        # sees the old file or the new one, never a truncated one.
        tmp_path = cache_path + ".part"
        quantized.tofile(tmp_path)
        os.replace(tmp_path, cache_path)
        del quantized
    except Exception:
        pass                                # a missed cache write is not fatal
    return cleaned


def _vad_cache_path(path):
    stat = os.stat(path)
    key = hashlib.sha1(
        f"{path}|{stat.st_size}|{stat.st_mtime}|vad|{SAMPLE_RATE}".encode("utf-8")
    ).hexdigest()
    directory = settings.cache_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, key + ".vad.json")


def _load_vad_cache(cache_path):
    """The raw (pre-duration-clip) Silero intervals for a file, or None on
    any cache miss/corruption - a missed read just means recomputing."""
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [(float(s), float(e)) for s, e in data]
    except Exception:
        return None


def _save_vad_cache(cache_path, intervals):
    """Same atomic tmp+os.replace pattern as _cleaned_cache_path's writer,
    so a reader on another analysis thread never sees a partial file."""
    try:
        tmp_path = cache_path + ".part"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(intervals, fh)
        os.replace(tmp_path, cache_path)
    except Exception:
        pass                                # a missed cache write is not fatal


def _gate(levels_db, hop_seconds):
    """Frame levels -> (start, end) speech intervals, via an adaptive gate."""
    if levels_db.size == 0:
        return [], -90.0, -90.0

    floor_db = float(np.percentile(levels_db, 10))
    speech_db = float(np.percentile(levels_db, 95))

    open_db = floor_db + THRESHOLD_FRACTION * max(speech_db - floor_db, 0.0)
    open_db = max(open_db, floor_db + MIN_MARGIN_DB, ABSOLUTE_FLOOR_DB,
                  speech_db - BELOW_SPEECH_DB)
    close_db = open_db - RELEASE_DB

    intervals = []
    start = None
    for i, level in enumerate(levels_db):
        if start is None:
            if level >= open_db:
                start = i
        elif level < close_db:
            intervals.append((start * hop_seconds,
                              (i + FRAME_SECONDS / hop_seconds) * hop_seconds))
            start = None
    if start is not None:
        intervals.append((start * hop_seconds,
                          (levels_db.size + FRAME_SECONDS / hop_seconds) * hop_seconds))

    intervals = [(max(0.0, s - ONSET_GUARD_SECONDS), e + ONSET_GUARD_SECONDS)
                 for s, e in intervals]
    intervals = _merge_close(intervals, HANGOVER_SECONDS)
    intervals = [(s, e) for s, e in intervals if e - s >= MIN_SPEECH_SECONDS]
    return intervals, floor_db, open_db


def _merge_close(intervals, max_gap):
    """Joins intervals separated by less than `max_gap`."""
    if not intervals:
        return []
    out = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [tuple(iv) for iv in out]


def speaking_intervals(path, denoiser_path=None, duration=None, log=None,
                       with_levels=False, with_samples=False):
    """
    Returns (start, end) ranges where this speaker is talking, found by
    Silero VAD (see silero_vad_onnx) on the RAW decoded audio - not the
    denoised/compressed/normalized copy. The old energy gate held "speaking"
    open through a natural trailing decay after someone stopped talking
    (confirmed by direct testing: compression flattens a decay tail's
    dynamic range, keeping it above threshold for most of a real pause), so
    cuts landed as a sliver right before the next speaker resumed instead of
    covering the whole gap. A neural VAD judges speech-likeness directly and
    isn't fooled by a merely-quieter-but-still-decaying tail the same way.
    Falls back to the old energy gate if Silero is unavailable for any
    reason (e.g. onnxruntime missing) - degraded, not broken.

    `denoiser_path` is rnnoise; used only for the cleaned copy below, not for
    Silero VAD detection itself.

    With `with_levels=True` returns (intervals, levels_db, hop) instead. The
    per-frame levels (still measured from the cleaned/denoised copy) are what
    lets auto-mute compare one speaker against another - deciding who is
    talking from a single microphone in isolation cannot tell a real voice
    from the other person bleeding into it.

    With `with_samples=True` the cleaned copy (see `clean_for_analysis`) comes
    back too, so a caller that also wants to draw the waveform from it - see
    app.py - doesn't have to decode and clean the file a second time.

    None of this touches the file or the audio the app plays and exports - it
    shapes a throwaway copy used to decide. Deliberately quiet: only genuine
    problems (like rnnoise or Silero being unavailable) reach `log`.
    """
    work = cleaned_samples_for(path, denoiser_path, log=log)
    levels_db, hop = _frame_levels(work)

    try:
        import silero_vad_onnx
        # Raw (pre-duration-clip) intervals, disk-cached by content hash -
        # Silero's ~112k sequential inference calls for a 1-hour track is
        # the slowest single step in analysis, and unlike the PCM/cleaned-
        # audio caches below, this result used to be recomputed on every
        # app session even for a file that hadn't changed since the last
        # one. Cached pre-clip (not keyed on `duration`) so it stays valid
        # across a shared-timeline duration change caused by adding or
        # removing an unrelated track.
        vad_cache_path = _vad_cache_path(path)
        intervals = _load_vad_cache(vad_cache_path)
        if intervals is None:
            # The memmap, not _load's float32 copy of it: the resample reads
            # this a block at a time and scales as it goes, so the only
            # full-length array anyone allocates is the 16kHz one the model
            # actually needs - a third the length and the one that gets
            # pickled to the child anyway. _load here cost 2.76GB on a
            # 2-hour track before the VAD had even started.
            raw = np.memmap(decode_to_pcm(path), dtype=np.int16, mode="r")
            intervals = silero_vad_onnx.speaking_intervals(raw, SAMPLE_RATE)
            _save_vad_cache(vad_cache_path, intervals)
    except Exception as exc:
        if log:
            log(f"  Silero VAD unavailable ({exc}); falling back to the energy gate")
        intervals, _floor_db, _gate_db = _gate(levels_db, hop)

    if duration:
        intervals = [(max(0.0, s), min(duration, e)) for s, e in intervals
                     if s < duration]

    result = [intervals]
    if with_levels:
        result += [levels_db, hop]
    if with_samples:
        result.append(work)
    return result[0] if len(result) == 1 else tuple(result)
