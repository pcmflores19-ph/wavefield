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

import numpy as np

from player import SAMPLE_RATE, decode_to_pcm

import bundled

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


def _load_track(path):
    pcm_path = decode_to_pcm(path)
    samples = np.memmap(pcm_path, dtype=np.int16, mode="r")
    return np.asarray(samples, dtype=np.float32) / 32768.0


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
        offline = chain.snapshot()
        processed = offline.process(audio, SAMPLE_RATE, reset=True)
        if processed.size == audio.size:
            audio = processed

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

    pieces = []
    for start, end in keep_ranges:
        a = max(0, int(start * SAMPLE_RATE))
        b = min(audio.size, int(end * SAMPLE_RATE))
        if b > a:
            pieces.append(audio[a:b])
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def write_wav(path, audio, sample_rate=SAMPLE_RATE):
    """Writes mono float32 (-1..1) as a 16-bit PCM WAV."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
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
    mutes = mutes or []
    written = []
    rendered = []

    for i, path in enumerate(speaker_paths):
        track_mutes = [(s, e) for lane, s, e in mutes if lane == i]
        chain = chains[i] if chains and i < len(chains) else None
        gain = gains[i] if gains and i < len(gains) else 1.0
        rendered.append(render_track(path, keep_ranges, track_mutes, chain, gain,
                                     progress=progress))

    if not rendered:
        raise ValueError("Nothing to render.")

    length = max(a.size for a in rendered)
    mix = np.zeros(length, dtype=np.float32)
    for audio in rendered:
        mix[:audio.size] += audio

    peak = float(np.abs(mix).max()) if mix.size else 0.0
    if peak > 1.0:
        # Summing speakers can overshoot; scale back rather than clip.
        if progress:
            progress(f"mix peaked at {peak:.2f}, scaling to avoid clipping")
        mix /= peak

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
    if len(bookends) > 1:
        mix = np.concatenate(bookends)

    if progress:
        progress(f"writing {os.path.basename(out_path)}")
    written.append(write_wav(out_path, mix))

    if stems:
        base, ext = os.path.splitext(out_path)
        for i, audio in enumerate(rendered):
            name = os.path.splitext(os.path.basename(speaker_paths[i]))[0]
            stem_path = f"{base}_{name}{ext or '.wav'}"
            if progress:
                progress(f"writing stem {os.path.basename(stem_path)}")
            written.append(write_wav(stem_path, audio))

    return written, peak


# --------------------------------------------------------------- baked-FX media

def bake_processed_media(speaker_paths, out_dir, mutes=None, chains=None,
                         gains=None, progress=None):
    """
    Writes <name>_processed.mp4 per speaker: the original video stream copied
    untouched, with the audio replaced by the VST-processed, muted, gain-staged
    version.

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

        out_path = os.path.join(out_dir, base + "_processed.mp4")
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-i", path,            # original: video taken from here
            "-i", wav_path,        # processed audio
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",        # no re-encode, no quality loss, fast
            "-c:a", "aac", "-b:a", "192k",
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
