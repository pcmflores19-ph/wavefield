"""
Renders the finished audio: cuts applied, regions muted, VST chains baked in.

For an audio-only podcast this is the whole deliverable - no round trip through
Resolve at all.

Processing order matters and mirrors what you hear while monitoring:
  1. VST chain over the speaker's FULL continuous recording, so time-dependent
     plugins (levelers, compressors, gates) see a natural signal rather than
     disjointed fragments,
  2. then mutes (silence is silence, whatever a plugin did),
  3. then the fader gain,
  4. and only then are the keep ranges extracted and concatenated.
"""

import os
import subprocess
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from player import SAMPLE_RATE, decode_to_pcm

import bundled
import effects

# Ramp length at each mute edge - long enough to kill the click,
# short enough to be inaudible as a fade.
MUTE_FADE_SECONDS = 0.010

FFMPEG = bundled.tool("ffmpeg")
FFPROBE = bundled.tool("ffprobe")


def decode_audio_file(path):
    """
    Decodes any audio/video file to mono float32 at SAMPLE_RATE. Used for the
    intro/outro beds, which are dropped in as-is - no cuts, mutes or VST
    processing are applied to them.
    """
    cmd = [
        FFMPEG, "-v", "error", "-i", path,
        "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not decode {os.path.basename(path)}:\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


# How much audio is converted or written at a time where a whole-track
# temporary would otherwise be built. ~4.2M samples, 16.8MB of float32.
_STREAM_BLOCK = 1 << 22


def _load_track(path):
    """The track as mono float32, filled a block at a time.

    `np.asarray(memmap, float32) / 32768.0` reads the same but costs two
    full-length arrays - the cast, then the divide's result - which on a
    2-hour track is 2.76GB before anything has been rendered. Scaling by
    2**-15 is exact, so this is bit-for-bit the same audio.
    """
    samples = np.memmap(decode_to_pcm(path), dtype=np.int16, mode="r")
    count = samples.shape[0]
    audio = np.empty(count, dtype=np.float32)
    for start in range(0, count, _STREAM_BLOCK):
        stop = min(start + _STREAM_BLOCK, count)
        audio[start:stop] = samples[start:stop]      # int16 -> float32, exact
        audio[start:stop] *= 1.0 / 32768.0
    return audio


def render_track(path, keep_ranges, mute_ranges=None, chain=None, gain=1.0,
                 progress=None):
    """Returns the finished mono float32 audio for one speaker."""
    if progress:
        progress(f"decoding {os.path.basename(path)}")
    audio = _load_track(path)

    if chain is not None and chain.active_slots():
        if progress:
            progress(f"processing {os.path.basename(path)} through {chain.describe()}")
        # Detached copy - see TrackChain.snapshot(). Rendering through the live
        # plugins while playback is running crashes the process.
        offline = chain.snapshot(log=progress)
        processed = offline.process(audio, SAMPLE_RATE, reset=True,
                                    log=progress)
        if processed.size == audio.size:
            audio = processed
        elif progress:
            # process() already lines lengths back up, so this should not
            # happen - but an export quietly containing none of the effects is
            # exactly the failure this whole path used to have, so say so.
            progress(f"WARNING: {os.path.basename(path)} came back "
                     f"{processed.size} samples for {audio.size}; the effects "
                     "chain was NOT applied to this track")

    # Ramp into and out of every mute. Cutting straight to zero puts a step
    # in the waveform, and a step is a click - audible on every single mute,
    # of which auto-mute makes hundreds.
    fade = max(1, int(MUTE_FADE_SECONDS * SAMPLE_RATE))
    for start, end in (mute_ranges or []):
        a = max(0, int(start * SAMPLE_RATE))
        b = min(audio.size, int(end * SAMPLE_RATE))
        if b <= a:
            continue
        # A short mute must not fade for longer than it lasts.
        span = min(fade, (b - a) // 2) or 1
        audio[a:a + span] *= np.linspace(1.0, 0.0, span, dtype=np.float32)
        if b - span > a + span:
            audio[a + span:b - span] = 0.0
        audio[b - span:b] *= np.linspace(0.0, 1.0, span, dtype=np.float32)

    if gain != 1.0:
        audio *= gain

    spans = []
    total = 0
    for start, end in keep_ranges:
        a = max(0, int(start * SAMPLE_RATE))
        b = min(audio.size, int(end * SAMPLE_RATE))
        if b > a:
            spans.append((a, b))
            total += b - a
    if not spans:
        return np.zeros(0, dtype=np.float32)

    kept = np.empty(total, dtype=np.float32)
    at = 0
    for a, b in spans:
        kept[at:at + (b - a)] = audio[a:b]
        at += b - a
    return kept


_DEFAULT_RENDER_WORKERS = 2


def render_tracks(speaker_paths, keep_ranges, mutes=None, chains=None,
                  gains=None, progress=None, max_workers=_DEFAULT_RENDER_WORKERS,
                  want_audio=False):
    """
    The summed, processed, cut mix across every speaker - shared by
    `export_audio` and `app._render_mix` so the two don't drift.

    Each track's decode+VST+effects chain runs in its own worker thread -
    safe because TrackChain.snapshot() (see its docstring) gives every track
    fully independent plugin instances, and pedalboard releases the GIL
    during process() (measured 2026-09-12), so this is real parallelism, not
    just overlapped I/O. `_ProcessLoadGate` already serializes the one
    genuinely unsafe part (loading a plugin) while letting processing passes
    run concurrently - this is exactly the case it was built for.

    Still resolved and summed in original track order, one at a time: the
    shared `mix` buffer and floating-point summation order must stay
    deterministic, so only the expensive per-track render is parallel.

    `max_workers` is capped low on purpose. Rendering N tracks at once holds
    N full tracks in memory instead of releasing each before the next -
    multi-hour tracks have already caused native OOM crashes at ~5GB for a
    single sequential pass, so this trades a bounded, predictable memory
    increase for wall-clock speed rather than scaling with CPU count.

    mutes:  [(speaker_index, start, end)]
    chains: [TrackChain or None] per speaker
    gains:  [float] per speaker
    want_audio: also return [(path, audio)] per speaker, for stem writing.

    Returns (mix, rendered) - `rendered` is [] unless want_audio is set.
    """
    mutes = mutes or []
    total = len(speaker_paths)
    workers = max(1, min(max_workers, total))

    def render_one(i, path):
        if progress:
            progress(f"Rendering {os.path.basename(path)}...")
        track_mutes = [(s, e) for lane, s, e in mutes if lane == i]
        chain = chains[i] if chains and i < len(chains) else None
        gain = gains[i] if gains and i < len(gains) else 1.0
        return render_track(path, keep_ranges, track_mutes, chain, gain,
                            progress=progress)

    mix = None
    rendered = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # At most `workers` renders in flight at once - submitting every
        # track upfront would let a fast later track finish and sit on its
        # (large) result while an earlier, slower one is still consumed in
        # order, letting more than `workers` tracks' audio be alive at once.
        pending = deque()
        submitted = 0
        while submitted < workers:
            pending.append(pool.submit(render_one, submitted,
                                       speaker_paths[submitted]))
            submitted += 1
        for i in range(total):
            path = speaker_paths[i]
            audio = pending.popleft().result()
            if submitted < total:
                pending.append(pool.submit(render_one, submitted,
                                           speaker_paths[submitted]))
                submitted += 1
            if want_audio:
                rendered.append((path, audio))
            if mix is None:
                mix = np.zeros(audio.size, dtype=np.float32)
            elif audio.size > mix.size:
                grown = np.zeros(audio.size, dtype=np.float32)
                grown[:mix.size] = mix
                mix = grown
            mix[:audio.size] += audio

    if mix is None:
        mix = np.zeros(0, dtype=np.float32)
    return mix, rendered


def write_wav(path, audio, sample_rate=SAMPLE_RATE):
    """
    Writes mono float32 (-1..1) as a 16-bit PCM WAV.

    `audio` may be one array or a sequence of them, written back to back -
    that is how the intro/outro beds go on without concatenating them onto
    the mix first. Converted and written a block at a time: building the
    clipped copy, the scaled copy, the int16 copy and then `tobytes()` of the
    whole thing cost ~4.8GB on a 2-hour mix. The arithmetic is element-wise,
    so the bytes are identical either way.
    """
    parts = audio if isinstance(audio, (list, tuple)) else (audio,)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for part in parts:
            for start in range(0, part.size, _STREAM_BLOCK):
                block = np.clip(part[start:start + _STREAM_BLOCK], -1.0, 1.0)
                block *= 32767.0
                w.writeframes(block.astype(np.int16).tobytes())
    return path


def export_audio(out_path, speaker_paths, keep_ranges, mutes=None, chains=None,
                 gains=None, stems=False, intro_path=None, outro_path=None,
                 progress=None):
    """
    Renders every speaker and writes either a single mixdown (default) or one
    stem per speaker alongside it.

    mutes:  [(speaker_index, start, end)]
    chains: [TrackChain or None] per speaker
    gains:  [float] per speaker
    intro_path / outro_path: audio dropped in front of / after the episode,
        untouched by cuts, mutes and VSTs. Mixdown only - stems stay clean.

    Returns (written_paths, peak_before_limiting).
    """
    written = []
    stem_paths = []
    base, ext = os.path.splitext(out_path)

    mix, rendered = render_tracks(speaker_paths, keep_ranges, mutes=mutes,
                                  chains=chains, gains=gains,
                                  progress=progress, want_audio=stems)

    if stems:
        for path, audio in rendered:
            name = os.path.splitext(os.path.basename(path))[0]
            stem_path = f"{base}_{name}{ext or '.wav'}"
            if progress:
                progress(f"writing stem {os.path.basename(stem_path)}")
            stem_paths.append(write_wav(stem_path, audio))

    if mix is None or mix.size == 0:
        raise ValueError("Nothing to render.")

    peak = 0.0
    for start in range(0, mix.size, _STREAM_BLOCK):
        block_peak = float(np.abs(mix[start:start + _STREAM_BLOCK]).max())
        peak = max(peak, block_peak)
    if peak > 1.0:
        # Summing speakers can overshoot; limit the loud moments instead of
        # turning the whole episode down for one overlap - matches what
        # live monitoring now does, so playback and the export sound alike.
        if progress:
            progress(f"mix peaked at {peak:.2f}, limiting overs")
        mix = effects.limiter(mix, SAMPLE_RATE, threshold_db=0.0)

    # Intro/outro go on last, at their own level, so scaling the episode mix
    # never changes how the music sounds.
    bookends = []
    if intro_path:
        if progress:
            progress(f"adding intro {os.path.basename(intro_path)}")
        bookends.append(decode_audio_file(intro_path))
    bookends.append(mix)
    if outro_path:
        if progress:
            progress(f"adding outro {os.path.basename(outro_path)}")
        bookends.append(decode_audio_file(outro_path))
    if progress:
        progress(f"writing {os.path.basename(out_path)}")
    # Handed over as parts and written back to back, rather than concatenated
    # into one array first - the WAV frames come out the same either way.
    written.append(write_wav(out_path, bookends))

    # Stems go on the end of the list so the order callers report is still
    # mixdown-then-stems.
    written.extend(stem_paths)

    return written, peak


# --------------------------------------------------------------- baked-FX media

def bake_processed_media(speaker_paths, out_dir, mutes=None, chains=None,
                         gains=None, progress=None):
    """
    Writes <name>_processed.mov per speaker: the original video stream copied
    untouched, with the audio replaced by the VST-processed, muted, gain-staged
    version.

    MOV rather than MP4 because the audio is copied in as 16-bit PCM: this is
    master material heading for a grade and a mix, so it is never re-encoded.
    MP4 cannot carry PCM, and it also refuses video codecs MOV accepts, so
    copying a ProRes or DNxHR source into it fails outright.

    This was briefly switched to MP4/AAC while chasing a "Media Offline" import,
    which turned out to be the source timecode (see fcpxml_writer) and nothing
    to do with the container. Changing it again will not fix a linking problem.

    Deliberately NOT cut - the audio stays full length and frame-aligned with
    the video, so the FCPXML can keep using exactly the same in/out points and
    the proven lane structure is unchanged. It just points at these files
    instead of the originals, which is what stops the effects work having to be
    repeated in Resolve.
    """
    mutes = mutes or []
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for i, path in enumerate(speaker_paths):
        duration = _probe_duration(path)
        track_mutes = [(s, e) for lane, s, e in mutes if lane == i]
        chain = chains[i] if chains and i < len(chains) else None
        gain = gains[i] if gains and i < len(gains) else 1.0

        if progress:
            progress(f"baking {os.path.basename(path)}")
        audio = render_track(path, [(0.0, duration)], track_mutes, chain, gain,
                             progress=progress)

        base = os.path.splitext(os.path.basename(path))[0]
        wav_path = os.path.join(out_dir, base + "_processed.wav")
        write_wav(wav_path, audio)

        out_path = os.path.join(out_dir, base + "_processed.mov")
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-i", path,            # original: video taken from here
            "-i", wav_path,        # processed audio
            # "?" so an audio-only recording bakes too rather than failing.
            "-map", "0:v:0?", "-map", "1:a:0",
            "-c:v", "copy",        # no re-encode, no quality loss, fast
            "-c:a", "copy",        # already 16-bit PCM; MOV carries it as-is
            "-shortest",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed muxing {os.path.basename(path)}:\n"
                f"{result.stderr.decode(errors='replace')}"
            )
        try:
            os.remove(wav_path)
        except OSError:
            pass
        written.append(out_path)

    return written


def _probe_duration(path):
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
    return float(result.stdout.strip())
