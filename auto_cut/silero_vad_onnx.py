"""
Torch-free Silero VAD inference: loads the ONNX model directly via
onnxruntime and reimplements get_speech_timestamps' windowing/hysteresis
logic in plain numpy.

Why not `pip install silero-vad`: its own package hard-requires torch and
torchaudio as core dependencies (not optional, even on the "onnx" model
path) - `utils_vad.py`'s own ONNX wrapper still takes torch Tensors as its
calling convention throughout (`x.dim()`, `x.unsqueeze(0)`, ...). Bundling
torch was already rejected for WhisperX specifically because of its
multi-GB weight vs. a ~100MB installer; the same reasoning applies here.
The MODEL ITSELF (silero_vad.onnx, ~2.3MB) has no such dependency - it's a
plain ONNX graph - so this module talks to it directly with onnxruntime and
numpy only.

Model: Silero VAD (github.com/snakers4/silero-vad), MIT licensed,
Copyright (c) 2020-present Silero Team. Bundled at assets/silero_vad.onnx.
The windowing/hysteresis logic below follows the same shape as their own
get_speech_timestamps (window_size_samples, threshold, neg_threshold,
padding) as a reference for the model's calling convention, not copied
code - every function in their reference implementation is torch-coupled.
"""

import multiprocessing
import os
import tempfile

import numpy as np
import onnxruntime as ort

import bundled
import settings

# Seen in practice (autocut_crash.log, 2026-09-11): onnxruntime's CPU
# execution provider can raise a native Windows fault - "int divide by
# zero" or "access violation" - out of session.run() on some inputs, on a
# long (~1 hour) multi-speaker session. That is not a Python exception; no
# try/except anywhere in this process can catch it, and it takes the whole
# GUI down with it. speaking_intervals() below already has a
# try/except-and-fall-back-to-the-energy-gate path for exactly this kind of
# failure (voice_activity.speaking_intervals) - it just can never be reached
# for a fault like this, because the process dies before Python's exception
# machinery runs. Isolating the actual session.run() call in a child process
# means a fault there kills only the child; the parent sees a normal,
# catchable failure and takes the existing fallback path instead of crashing.
_WORKER_TIMEOUT_SECONDS = 300.0

# The model is trained on exactly these chunk sizes; anything else is
# silently wrong (the model was not trained on other window lengths).
_WINDOW_SAMPLES = {8000: 256, 16000: 512}

# This model version (6.2.1) prepends a lookback buffer from the tail of the
# PREVIOUS chunk to each window before inference - without it, every chunk
# is evaluated with no temporal context and the model never gets confident
# enough to cross the speech threshold (confirmed by direct comparison
# against the reference OnnxWrapper on identical audio: without context,
# probabilities topped out around 0.13 on real speech; with it, correctly
# reached >0.99). Not discoverable from the ONNX graph's I/O shapes alone
# (`input [None, None]`) - only from the reference implementation's own
# call sequence.
_CONTEXT_SAMPLES = {8000: 32, 16000: 64}

_session = None


def _get_session():
    global _session
    if _session is None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(
            bundled.asset("silero_vad.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _session


# Output samples converted per pass. At 16kHz this is ~65s of audio per block
# (~4MB of float32, plus ~12MB for the 48kHz span it reads from), which is
# nothing next to a track and big enough that the loop overhead disappears.
_RESAMPLE_BLOCK = 1 << 20


def _resample(samples, orig_rate, target_rate):
    """
    Linear-interpolation resample - VAD doesn't need broadcast quality, just
    to land on a rate the model accepts (8000 or 16000 Hz).

    Done a block at a time, and `samples` may be an int16 np.memmap straight
    off the decode cache (it is scaled here, so the caller never has to
    materialise a float32 copy of the whole track). The obvious version -
    np.linspace over the full length, then np.interp - cost ~7.8GB of float64
    temporaries on a 2-hour track, because linspace is float64 and interp
    upcasts the samples to match: the single largest allocation in the app,
    on a path that runs on every Analyze. See autocut_crash.log.
    """
    count = samples.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.float32)

    # int16 off the decode cache scales to -1..1; float input is already there.
    scale = (1.0 / 32768.0) if np.issubdtype(samples.dtype, np.integer) else None

    if orig_rate == target_rate:
        out = np.empty(count, dtype=np.float32)
        for start in range(0, count, _RESAMPLE_BLOCK):
            stop = min(start + _RESAMPLE_BLOCK, count)
            block = np.asarray(samples[start:stop], dtype=np.float32)
            if scale is not None:
                block *= scale
            out[start:stop] = block
        return out

    duration = count / orig_rate
    target_count = max(0, int(round(duration * target_rate)))
    if target_count <= 0:
        return np.zeros(0, dtype=np.float32)

    # Output j reads source position j * ratio - the same mapping the old
    # linspace pair described, without building either axis.
    ratio = count / target_count
    out = np.empty(target_count, dtype=np.float32)

    for start in range(0, target_count, _RESAMPLE_BLOCK):
        stop = min(start + _RESAMPLE_BLOCK, target_count)
        pos = np.arange(start, stop, dtype=np.float64) * ratio
        left = np.floor(pos).astype(np.int64)
        np.clip(left, 0, count - 1, out=left)
        right = np.minimum(left + 1, count - 1)

        # One contiguous read covering this block, rather than a gather per
        # sample - `left` is monotonic, so the span is exactly what's needed.
        lo = int(left[0])
        hi = int(right[-1]) + 1
        span = np.asarray(samples[lo:hi], dtype=np.float32)
        if scale is not None:
            span *= scale

        a = span[left - lo]
        b = span[right - lo]
        frac = (pos - left).astype(np.float32)
        out[start:stop] = a + (b - a) * frac

    return out


def speech_probabilities(samples, sample_rate=16000):
    """
    Mono float32 samples at 8000 or 16000 Hz -> per-window speech
    probability (one value per _WINDOW_SAMPLES[sample_rate]-sample chunk).

    Not crash-safe: this is the function that faulted natively in
    autocut_crash.log ("int divide by zero" / "access violation" inside
    onnxruntime, on some inputs on long sessions). Application code should
    call speech_probabilities_isolated() instead, which runs this same call
    in a child process so a fault there can't take the whole app down. This
    function exists as the isolated version's actual worker body and for
    direct testing - it is not meant to be called from the app itself.
    """
    if sample_rate not in _WINDOW_SAMPLES:
        raise ValueError(f"sample_rate must be 8000 or 16000, got {sample_rate}")
    window = _WINDOW_SAMPLES[sample_rate]
    context_size = _CONTEXT_SAMPLES[sample_rate]
    session = _get_session()

    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(sample_rate, dtype=np.int64)
    context = np.zeros((1, context_size), dtype=np.float32)

    probs = []
    total = samples.shape[0]
    for start in range(0, max(total, 1), window):
        chunk = samples[start:start + window]
        if chunk.shape[0] == 0:
            break
        if chunk.shape[0] < window:
            chunk = np.pad(chunk, (0, window - chunk.shape[0]))
        chunk = chunk.reshape(1, -1).astype(np.float32)
        model_input = np.concatenate([context, chunk], axis=1)
        output, state = session.run(
            ["output", "stateN"],
            {"input": model_input, "state": state, "sr": sr},
        )
        context = model_input[:, -context_size:]
        probs.append(float(output[0, 0]))
    return np.array(probs, dtype=np.float32)


def _speech_probabilities_worker(input_path, count, sample_rate, conn):
    """
    Runs in the child process, reading the (already resampled) track from a
    disk-backed memmap rather than through `args` - see
    speech_probabilities_isolated for why. Sends ("ok", array) or
    ("error", message); the returned probabilities array is tiny (one float
    per ~32ms window) so it still goes back over the pipe directly.
    """
    try:
        samples = np.memmap(input_path, dtype=np.float32, mode="r",
                            shape=(count,))
        result = speech_probabilities(samples, sample_rate)
        conn.send(("ok", result))
    except Exception as exc:
        conn.send(("error", str(exc)))
    finally:
        conn.close()


def speech_probabilities_isolated(samples, sample_rate=16000):
    """
    Same contract and return value as speech_probabilities(), but runs the
    actual onnxruntime call in a child process - see the module comment on
    _WORKER_TIMEOUT_SECONDS for why. A crashed or hung child is reported as a
    normal RuntimeError, which speaking_intervals()'s caller
    (voice_activity.speaking_intervals) already catches and falls back to the
    energy gate for.

    `samples` (the resampled 16kHz track, up to ~230MB/hour) crosses into the
    child via a disk-backed memmap, not `Process(args=...)` - passing the
    array directly pickles it across the Windows spawn boundary, which for a
    moment holds parent + pickle buffer + child copies at once, per speaker.
    That is what actually crashed/froze long multi-speaker sessions.
    """
    directory = settings.cache_dir()
    os.makedirs(directory, exist_ok=True)
    fd, in_path = tempfile.mkstemp(dir=directory, suffix=".vad_in.pcm")
    os.close(fd)
    try:
        count = samples.shape[0]
        memmap_in = np.memmap(in_path, dtype=np.float32, mode="w+",
                              shape=(count,))
        block = _RESAMPLE_BLOCK
        for start in range(0, count, block):
            stop = min(start + block, count)
            memmap_in[start:stop] = samples[start:stop]
        memmap_in.flush()
        del memmap_in
        # Unlike the denoise path (voice_activity._denoise_isolated), there is
        # no wrapper function between this and its caller holding a second
        # reference to `samples` - so dropping it here actually frees the
        # array in the parent for the full duration of the child's run below,
        # not just a brief window.
        del samples

        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        proc = multiprocessing.Process(
            target=_speech_probabilities_worker,
            args=(in_path, count, sample_rate, child_conn), daemon=True)
        proc.start()
        child_conn.close()  # only the child should hold the writable end
        try:
            if not parent_conn.poll(_WORKER_TIMEOUT_SECONDS):
                proc.terminate()
                proc.join(5.0)
                raise RuntimeError("Silero VAD worker timed out")
            try:
                status, payload = parent_conn.recv()
            except EOFError:
                # The pipe closed with nothing sent - the child died before
                # replying, which for a daemon whose only job is
                # send-then-exit means a crash (a fatal native fault, most
                # likely), not a clean unhandled Python exception (that path
                # already sends "error").
                proc.join(5.0)
                raise RuntimeError(
                    f"Silero VAD worker crashed (exit code {proc.exitcode})")
        finally:
            parent_conn.close()
        proc.join(5.0)
        if status == "ok":
            return payload
        raise RuntimeError(f"Silero VAD failed in worker process: {payload}")
    finally:
        try:
            if in_path and os.path.exists(in_path):
                os.remove(in_path)
        except OSError:
            pass


def _merge_close(intervals, max_gap):
    if not intervals:
        return []
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(iv) for iv in out]


def intervals_from_probabilities(probs, window_seconds,
                                 threshold=0.5, neg_threshold=None,
                                 padding_seconds=0.1,
                                 min_speech_seconds=0.2,
                                 hangover_seconds=0.15):
    """
    Per-window speech probabilities -> merged (start, end) second intervals.

    Same hysteresis shape as Silero's own get_speech_timestamps (a speech
    region starts once probability crosses `threshold`, and only ends once
    it drops below the lower `neg_threshold` - not the same value - so an
    ordinary mid-word dip in the model's own confidence doesn't split one
    word into two), reimplemented in plain numpy since every reference
    implementation is torch-coupled. This is a probability-domain gate, not
    an energy-domain one - a natural volume decay tail after speech ends
    does not itself keep the model confident that speech is still
    happening, which is the actual mechanism that broke the old RMS gate.
    """
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    intervals = []
    start = None
    for i, prob in enumerate(probs):
        if start is None:
            if prob >= threshold:
                start = i
        elif prob < neg_threshold:
            intervals.append((start * window_seconds, (i + 1) * window_seconds))
            start = None
    if start is not None:
        intervals.append((start * window_seconds, len(probs) * window_seconds))

    intervals = [(max(0.0, s - padding_seconds), e + padding_seconds)
                for s, e in intervals]
    intervals = _merge_close(intervals, hangover_seconds)
    return [(s, e) for s, e in intervals if e - s >= min_speech_seconds]


def speaking_intervals(samples, sample_rate, target_rate=16000, **kwargs):
    """
    End-to-end: mono float32 samples at any sample_rate -> (start, end)
    second intervals where Silero VAD detects speech. Matches
    voice_activity.speaking_intervals()'s output contract.
    """
    # Not bound to a local first: speech_probabilities_isolated drops its own
    # reference to this once it's written to the memmap (see its own
    # del samples), and binding it here too would keep the resampled array
    # alive for the whole isolated child's run for nothing.
    window = _WINDOW_SAMPLES[target_rate]
    probs = speech_probabilities_isolated(
        _resample(samples, sample_rate, target_rate), target_rate)
    return intervals_from_probabilities(probs, window / target_rate, **kwargs)
