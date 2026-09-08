"""
Waveform peaks and audio preview rendering, both via ffmpeg.

The waveform is the mix of every speaker, so the quiet stretches you see are
genuinely "nobody is talking" - the same thing the cut logic keys off.

The preview renders the *edited* audio (silences already removed) so you hear
what the export will sound like, not the raw recording.
"""

import os
import subprocess
import tempfile

import numpy as np

import bundled

FFMPEG = bundled.tool("ffmpeg")
PEAK_SAMPLE_RATE = 8000  # plenty for drawing; keeps decode fast

# Peaks are extracted once at this resolution and re-bucketed in the UI when
# zooming, so zooming never needs another decode. 50/s = 20ms per peak, fine
# down to word-level zoom, and only ~175k floats for a 58-minute episode.
PEAKS_PER_SECOND = 50


def _amix_filter(n_inputs):
    if n_inputs == 1:
        return "[0:a]anull"
    labels = "".join(f"[{i}:a]" for i in range(n_inputs))
    return f"{labels}amix=inputs={n_inputs}:duration=longest:normalize=0"


def _peaks_from_samples(samples, buckets):
    if samples.size == 0 or buckets <= 0:
        return np.zeros(max(buckets, 1), dtype=np.float32)
    per_bucket = int(np.ceil(samples.size / buckets))
    padded = np.zeros(per_bucket * buckets, dtype=np.float32)
    padded[: samples.size] = np.abs(samples)
    return padded.reshape(buckets, per_bucket).max(axis=1)


def extract_peaks(paths, buckets):
    """
    Peaks (0..1) of the mixed audio of `paths`, as one array of `buckets` values.
    """
    cmd = [FFMPEG, "-v", "error"]
    for p in paths:
        cmd += ["-i", p]
    cmd += [
        "-filter_complex", _amix_filter(len(paths)) + "[out]",
        "-map", "[out]",
        "-ac", "1", "-ar", str(PEAK_SAMPLE_RATE),
        "-f", "s16le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg peak extraction failed:\n{result.stderr.decode(errors='replace')}")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return _peaks_from_samples(samples, buckets)


def render_preview(paths, keep_ranges, start_seconds, preview_seconds=15.0):
    """
    Renders a short WAV of the EDITED audio (keep ranges concatenated), picking
    up at the first keep range at/after start_seconds and collecting roughly
    preview_seconds of kept audio.

    Returns the temp WAV path, or None if there's nothing to play.
    """
    segments = []
    collected = 0.0
    for seg_start, seg_end in keep_ranges:
        if seg_end <= start_seconds:
            continue
        s = max(seg_start, start_seconds)
        e = seg_end
        if e - s <= 0.01:
            continue
        if collected + (e - s) > preview_seconds:
            e = s + (preview_seconds - collected)
        segments.append((s, e))
        collected += e - s
        if collected >= preview_seconds:
            break

    if not segments:
        return None

    n = len(segments)
    parts = [_amix_filter(len(paths)) + "[mix]"]
    parts.append(f"[mix]asplit={n}" + "".join(f"[s{i}]" for i in range(n)))
    for i, (s, e) in enumerate(segments):
        parts.append(f"[s{i}]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]")
    parts.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]")

    out_path = os.path.join(tempfile.gettempdir(), "autocut_preview.wav")
    cmd = [FFMPEG, "-y", "-v", "error"]
    for p in paths:
        cmd += ["-i", p]
    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[out]",
        # winsound needs a plain PCM wav
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg preview render failed:\n{result.stderr.decode(errors='replace')}")
    return out_path


def peaks_from_samples(samples, buckets):
    """Peaks for audio already in memory - used for VST-processed waveforms."""
    return _peaks_from_samples(samples, buckets)


def processed_peaks(path, chain, duration_seconds, log=None,
                    peaks_per_second=PEAKS_PER_SECOND):
    """
    Peaks for one speaker: the real recording, with their VST chain on top.

    Deliberately the RAW file, not the denoised/levelled copy the analysis
    uses. The waveform has to be the audio - a picture normalized to -14 LUFS
    can never show clipping, and a picture already denoised makes the user's
    own denoiser look like it does nothing. What the analysis does to decide
    the cuts is its own business; see voice_activity.clean_for_analysis.

    Both the at-rest draw and the redraw after a chain edit come through here,
    so the two can never disagree.
    """
    import numpy as np
    from player import SAMPLE_RATE, decode_to_pcm

    pcm_path = decode_to_pcm(path)
    samples = np.asarray(np.memmap(pcm_path, dtype=np.int16, mode="r"),
                         dtype=np.float32) / 32768.0

    if chain is not None and chain.active_slots():
        # A detached copy: the live chain belongs to the audio callback, and
        # pushing an hour of audio through it from here would stall the audio
        # thread and drive the same VST from two threads at once.
        offline = chain.snapshot(log=log)
        processed = offline.process(samples, SAMPLE_RATE, reset=True)
        if processed.size == samples.size:
            samples = processed

    buckets = max(1, int(round(duration_seconds * peaks_per_second)))
    return _peaks_from_samples(samples, buckets)
