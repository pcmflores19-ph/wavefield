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
    result = subprocess.run(cmd, capture_output=True,
                            creationflags=getattr(subprocess,
                                                  "CREATE_NO_WINDOW", 0))
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
                 progress=None, offline_chain=None, should_cancel=None):
    """
    Returns the finished mono float32 audio for one speaker.

    `offline_chain`, if given, is an already-`snapshot()`ed detached chain to
    process through instead of snapshotting `chain` here - see
    `render_tracks`, which pre-snapshots every track's chain sequentially
    before parallelizing this function's decode+process work across tracks.
    Loading VST3 plugins must never happen concurrently (native crash risk;
    see TrackChain.snapshot()'s docstring and _ProcessLoadGate), so callers
    that render more than one track at once must snapshot up front, not here.

    `should_cancel`, if given, is passed into the VST/effects pass below -
    see TrackChain.process_slots() for the granularity it's checked at
    (between plugins, not mid-plugin).
    """
    if progress:
        progress(f"decoding {os.path.basename(path)}")
    audio = _load_track(path)

    offline = offline_chain
    if offline is None and chain is not None and chain.active_slots():
        # Detached copy - see TrackChain.snapshot(). Rendering through the
        # live plugins while playback is running crashes the process.
        offline = chain.snapshot(log=progress)

    if offline is not None:
        if progress:
            progress(f"processing {os.path.basename(path)} through {offline.describe()}")
        processed = offline.process(audio, SAMPLE_RATE, reset=True,
                                    log=progress, should_cancel=should_cancel)
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


# Below the "16GB" a machine is marketed/sold as, not exactly 16GB itself:
# GlobalMemoryStatusEx reports RAM actually usable by Windows, which is
# always somewhat less than nominal capacity (hardware/firmware
# reservations - integrated graphics, motherboard-reserved regions, etc.).
# Confirmed empirically: this project's own 16GB dev machine reports
# ~15.7GB (16896126976 bytes) here, which a literal 16*1024**3 threshold
# would have misclassified as "below 16GB" and silently dropped to the
# slower 1-worker path on the exact machine this was measured and
# confirmed fine on. 15GB leaves comfortable room below a real 16GB
# machine's reported total while still cleanly excluding real 8GB/12GB
# machines, which report well below it.
_RENDER_WORKERS_RAM_THRESHOLD_BYTES = 15 * (1024 ** 3)

# Tiers above the one confirmed by measurement (2 workers/15GB, ~11GB peak).
# Reasoned extrapolation, not calibrated fact - there is still only that
# one real data point. Each additional concurrent track has been costing
# roughly another ~5.5GB (11GB peak for 2 workers, against a lone
# sequential pass's own separately-measured ~5GB peak for one track), so
# these thresholds leave headroom above that estimate similar to what the
# confirmed tier leaves above its own. Revisit with real measurements if a
# user on one of these tiers reports trouble - see _auto_render_workers.
_RENDER_WORKERS_RAM_TIERS = (
    (48 * (1024 ** 3), 4),
    (32 * (1024 ** 3), 3),
    (_RENDER_WORKERS_RAM_THRESHOLD_BYTES, 2),
)


def _total_ram_bytes():
    """
    Total installed physical RAM, via the Windows API directly (this app is
    Windows-only today - see packaging/autocut.spec and README.md - so this
    avoids adding psutil as a new bundled dependency for something ctypes
    already covers, matching the existing convention of reaching for ctypes
    directly for Windows-specific needs elsewhere in this codebase, e.g.
    app.py's DPI awareness and vst_host.py's window z-order calls).

    Total, not "currently available" - available RAM swings with whatever
    else the user has open and isn't a stable per-machine signal, where
    total installed RAM is the actual ceiling this decision cares about.

    Returns None if the query fails for any reason (unusual Windows
    configuration, running under Wine, etc.) - this is a nice-to-have sizing
    heuristic, not something that should ever block an export.
    """
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys)
    except Exception:
        return None


def _auto_render_workers():
    """
    2 concurrent tracks costs real memory - each worker holds a full
    raw-length decode buffer (a 2-hour track alone is ~2.76GB before any
    processing, see render_track's docstring) - measured at ~11GB peak on a
    16GB machine after the transient-duplication fixes elsewhere in this
    file. That leaves too little headroom for the OS and whatever else is
    running (a browser, DaVinci Resolve itself) on a nominal-16GB machine
    or smaller, so machines below that threshold get the slower but safer
    1-worker path instead.

    Above that one confirmed tier, `_RENDER_WORKERS_RAM_TIERS` adds further
    tiers for machines with room to run more speaker tracks at once - most
    podcasts are 2-3 speakers anyway, and `render_tracks` already caps the
    actual worker count at the number of tracks that exist
    (`workers = max(1, min(max_workers, total))`), so this only ever raises
    the ceiling for shows with more hosts on hardware that can afford it.
    These higher tiers are reasoned extrapolation from the one real
    measurement, not additional confirmed data points - see the tiers'
    own comment. Revisit if reports from machines on these tiers come in,
    same as the original single-threshold design already asked for.

    Falls back to today's known-safe default (2) if RAM can't be
    determined at all, rather than guessing low and silently costing
    everyone speed for a query failure that isn't actually a memory
    problem.
    """
    total = _total_ram_bytes()
    if total is None:
        return 2
    for threshold, workers in _RENDER_WORKERS_RAM_TIERS:
        if total >= threshold:
            return workers
    return 1


_DEFAULT_RENDER_WORKERS = _auto_render_workers()


def render_tracks(speaker_paths, keep_ranges, mutes=None, chains=None,
                  gains=None, progress=None, max_workers=_DEFAULT_RENDER_WORKERS,
                  want_audio=False, on_track=None, should_cancel=None):
    """
    The summed, processed, cut mix across every speaker - shared by
    `export_audio` and `app._render_mix` so the two don't drift.

    Each track's decode+VST+effects processing runs in its own worker thread
    - safe because TrackChain.snapshot() (see its docstring) gives every
    track fully independent plugin instances, and pedalboard releases the
    GIL during process() (measured 2026-09-12), so this is real parallelism,
    not just overlapped I/O.

    Loading is a different story and stays fully sequential: every chain is
    snapshotted (which loads VST3 plugins onto the main thread) up front,
    one track at a time, before any parallel work starts. Concurrent VST3
    loads are a native crash risk, and _ProcessLoadGate only serializes them
    safely, not quickly - two tracks' worth of loads racing at once was
    measured to compound export's momentary UI freeze and, worse, could push
    a queued load past its own gate timeout, silently dropping that plugin
    from the track with no loud warning (2026-09-13). Only the actual
    process() calls - the expensive part, not the loading - run concurrently.

    Still resolved and summed in original track order, one at a time: the
    shared `mix` buffer and floating-point summation order must stay
    deterministic, so only the expensive per-track render is parallel.

    `max_workers` is capped low on purpose. Rendering N tracks at once holds
    N full tracks in memory instead of releasing each before the next -
    multi-hour tracks have already caused native OOM crashes at ~5GB for a
    single sequential pass, so this trades a bounded, predictable memory
    increase for wall-clock speed rather than scaling with CPU count. The
    default itself (see `_auto_render_workers`) is picked from the
    machine's total RAM, not a flat constant - 2 workers costs real memory
    (~11GB peak measured on a 16GB+ machine), too tight on anything smaller.

    mutes:  [(speaker_index, start, end)]
    chains: [TrackChain or None] per speaker
    gains:  [float] per speaker
    want_audio: also expose each speaker's rendered audio, for stem writing.
    on_track: if given (and want_audio is set), called as on_track(path,
        audio) once per track, in original track order, as each track's
        result is consumed here - never from inside a worker thread, since
        two workers calling it directly would deliver tracks in completion
        order rather than original order, silently reordering stems for any
        caller that cares. Lets a caller (export_audio) write each stem out
        and drop its audio immediately instead of this function holding
        every track's audio alive until the end, which used to defeat the
        "at most `workers` tracks' audio alive at once" memory bound this
        function otherwise provides. If not given, falls back to the old
        behavior of accumulating into the returned `rendered` list.

    `should_cancel`, if given, is passed into each track's render (so an
    in-flight VST chain stops between plugins - see
    TrackChain.process_slots()) and is also checked between tracks: once
    tripped, no further tracks are submitted to the pool, though up to
    `workers` already in flight still finish and are folded into the
    returned mix rather than discarded.

    Returns (mix, rendered) - `rendered` is [] whenever want_audio is unset
    or an on_track callback was used to consume the audio instead.
    """
    mutes = mutes or []
    total = len(speaker_paths)
    workers = max(1, min(max_workers, total))

    # Every plugin load happens here, one track at a time, before any
    # parallel work starts - see the docstring above for why loading must
    # never race across tracks even though processing safely can.
    offline_chains = []
    for i in range(total):
        chain = chains[i] if chains and i < len(chains) else None
        if chain is not None and chain.active_slots():
            offline_chains.append(chain.snapshot(log=progress))
        else:
            offline_chains.append(None)

    def render_one(i, path):
        if progress:
            progress(f"Rendering {os.path.basename(path)}...")
        track_mutes = [(s, e) for lane, s, e in mutes if lane == i]
        gain = gains[i] if gains and i < len(gains) else 1.0
        result = render_track(path, keep_ranges, track_mutes, gain=gain,
                              progress=progress,
                              offline_chain=offline_chains[i],
                              should_cancel=should_cancel)
        if progress:
            # An explicit, unambiguous "this track is done" signal - callers
            # that turn these messages into a real progress fraction (see
            # app.py's _track_render_progress) can't otherwise tell a track
            # actually finished apart from just having been mentioned in a
            # decode/processing message, which is what made the old
            # per-message fraction guesswork possible in the first place.
            progress(f"finished {os.path.basename(path)}")
        return result

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
        consumed = 0
        while pending:
            path = speaker_paths[consumed]
            audio = pending.popleft().result()
            consumed += 1
            if submitted < total and not (should_cancel and should_cancel()):
                pending.append(pool.submit(render_one, submitted,
                                           speaker_paths[submitted]))
                submitted += 1
            if want_audio:
                if on_track is not None:
                    on_track(path, audio)
                else:
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


def apply_master(mix, master_chain, progress=None, should_cancel=None):
    """
    Runs the master bus over the finished, summed mix and returns the result.

    Mirrors what a DAW's master bus does: it sees the sum of every track, after
    their own effects, and comes BEFORE the -1 dBTP safety limiter the callers
    apply next. Stems and the per-track media for Resolve never come through
    here - only the mixdown does, the same way a single track rendered from a
    DAW does not carry the master bus.

    One continuous pass over the whole mix on a detached snapshot, for the same
    reasons render_track does it that way (see TrackChain.snapshot and
    process_slots: live plugins must not be driven from a second thread, and
    latency-compensating plugins must not be fed in blocks). A master that
    cannot be applied hands the mix back untouched, with a loud message - an
    export quietly missing its master is the failure this is built to avoid.
    """
    if (master_chain is None or mix is None or mix.size == 0
            or not master_chain.active_slots()):
        return mix
    if should_cancel and should_cancel():
        return mix
    offline = master_chain.snapshot(log=progress)
    if progress:
        progress(f"mastering the mix through {offline.describe()}")
    processed = offline.process(mix, SAMPLE_RATE, reset=True, log=progress,
                                should_cancel=should_cancel)
    if processed.size == mix.size:
        return processed
    if progress:
        progress(f"WARNING: the master chain came back {processed.size} "
                 f"samples for {mix.size}; it was NOT applied to the mix")
    return mix


def limit_to_ceiling(audio):
    """
    Applies the same -1 dBTP safety ceiling (effects.LIMITER_CEILING_DB) the
    mix already gets, chunked the same way. Used for audio that bypasses the
    mix's own peak check - currently the intro/outro bookends, which are
    appended after the mix is already limited (see export_audio, and
    app.py's _export_video_worker) and would otherwise reach the downstream
    AAC mux unchecked.
    """
    peak = 0.0
    for start in range(0, audio.size, _STREAM_BLOCK):
        peak = max(peak, effects.true_peak(audio[start:start + _STREAM_BLOCK]))
    ceiling = 10 ** (effects.LIMITER_CEILING_DB / 20.0)
    if peak > ceiling:
        audio = effects.limiter(audio, SAMPLE_RATE, threshold_db=effects.LIMITER_CEILING_DB)
    return audio


def export_audio(out_path, speaker_paths, keep_ranges, mutes=None, chains=None,
                 gains=None, stems=False, intro_path=None, outro_path=None,
                 progress=None, should_cancel=None, master_chain=None):
    """
    Renders every speaker and writes either a single mixdown (default) or one
    stem per speaker alongside it.

    mutes:  [(speaker_index, start, end)]
    chains: [TrackChain or None] per speaker
    gains:  [float] per speaker
    master_chain: TrackChain run over the summed mix (mixdown only - stems stay
        exactly what each speaker's own chain produced)
    intro_path / outro_path: audio dropped in front of / after the episode,
        untouched by cuts, mutes and VSTs. Mixdown only - stems stay clean.

    `should_cancel`, if given, is passed into render_tracks (per-track and
    per-plugin granularity - see its docstring) and checked again once
    rendering returns: on a cancel, no output/stem files are written at all,
    matching the all-or-nothing behavior a render failure already has,
    rather than leaving a truncated mix on disk.

    Returns (written_paths, peak_before_limiting).
    """
    written = []
    stem_paths = []
    base, ext = os.path.splitext(out_path)

    # Stems are written to <final>.tmp as each track finishes rendering
    # (see on_track below) rather than held in memory until every track is
    # done - that used to make render_tracks' "at most `workers` tracks'
    # audio alive at once" bound not apply to stems export at all, since its
    # own `rendered` list kept every track's full audio alive regardless.
    # Written to temp paths and renamed to their final names only after
    # render_tracks returns successfully, so a mid-render failure still
    # leaves zero stem files behind - same all-or-nothing behavior as before
    # this change, just without holding every stem's audio at once to get
    # it. (Renaming N files is atomic per file, not as a set - a crash
    # between renames could leave a partial set on disk; that narrow window
    # is unchanged from the disk-write reality any exporter has, not
    # something this change claims to close.)
    tmp_stem_paths = []

    def on_track(path, audio):
        name = os.path.splitext(os.path.basename(path))[0]
        stem_path = f"{base}_{name}{ext or '.wav'}"
        tmp_path = stem_path + ".tmp"
        if progress:
            progress(f"writing stem {os.path.basename(stem_path)}")
        write_wav(tmp_path, audio)
        tmp_stem_paths.append((tmp_path, stem_path))

    try:
        mix, _rendered = render_tracks(speaker_paths, keep_ranges, mutes=mutes,
                                       chains=chains, gains=gains,
                                       progress=progress, want_audio=stems,
                                       on_track=on_track if stems else None,
                                       should_cancel=should_cancel)
    except BaseException:
        for tmp_path, _final_path in tmp_stem_paths:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        raise

    if should_cancel and should_cancel():
        for tmp_path, _final_path in tmp_stem_paths:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return [], 0.0

    # Before the stems are moved into place, so a cancel or failure here still
    # leaves no partial set of files behind - same all-or-nothing rule as above.
    try:
        mix = apply_master(mix, master_chain, progress=progress,
                           should_cancel=should_cancel)
    except BaseException:
        for tmp_path, _final_path in tmp_stem_paths:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        raise
    if should_cancel and should_cancel():
        for tmp_path, _final_path in tmp_stem_paths:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return [], 0.0

    for tmp_path, final_path in tmp_stem_paths:
        os.replace(tmp_path, final_path)
        stem_paths.append(final_path)

    if mix is None or mix.size == 0:
        raise ValueError("Nothing to render.")

    # True peak, not raw sample peak - catches inter-sample overshoot a
    # lossy re-encode downstream (video export's AAC mux) can expose even
    # when no sample here ever reads over 1.0. Chunked the same way the old
    # sample-peak loop was; true peak's interpolation only looks at
    # neighboring samples, so a chunk boundary costs at most one negligible,
    # bounded estimate, not a new correctness gap.
    peak = 0.0
    for start in range(0, mix.size, _STREAM_BLOCK):
        peak = max(peak, effects.true_peak(mix[start:start + _STREAM_BLOCK]))
    ceiling = 10 ** (effects.LIMITER_CEILING_DB / 20.0)
    if peak > ceiling:
        # Summing speakers can overshoot; limit the loud moments instead of
        # turning the whole episode down for one overlap - matches what
        # live monitoring now does, so playback and the export sound alike.
        # -1 dBTP ceiling (effects.LIMITER_CEILING_DB), not 0 dBFS, so a
        # downstream lossy re-encode has headroom for its own overshoot.
        if progress:
            progress(f"mix peaked at {peak:.2f}, limiting overs")
        mix = effects.limiter(mix, SAMPLE_RATE, threshold_db=effects.LIMITER_CEILING_DB)

    # Intro/outro go on last, at their own level, so scaling the episode mix
    # never changes how the music sounds. They still get the same -1 dBTP
    # safety ceiling as the mix, independently, so mastered music parked at
    # 0 dBFS doesn't clip on the same downstream AAC re-encode this ceiling
    # exists for - without touching their loudness relative to the mix.
    bookends = []
    if intro_path:
        if progress:
            progress(f"adding intro {os.path.basename(intro_path)}")
        bookends.append(limit_to_ceiling(decode_audio_file(intro_path)))
    bookends.append(mix)
    if outro_path:
        if progress:
            progress(f"adding outro {os.path.basename(outro_path)}")
        bookends.append(limit_to_ceiling(decode_audio_file(outro_path)))
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
                         gains=None, progress=None, should_cancel=None):
    """
    Writes <name>_processed.mov per speaker: the original video stream copied
    untouched, with the audio replaced by the VST-processed, muted, gain-staged
    version.

    `should_cancel`, if given, is checked once per speaker, before that
    speaker's render starts, and is also passed into render_track for
    per-plugin granularity within a speaker already in progress. Speakers
    already baked when cancellation is observed keep their output files;
    remaining speakers are simply skipped.

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
        if should_cancel and should_cancel():
            break

        duration = _probe_duration(path)
        track_mutes = [(s, e) for lane, s, e in mutes if lane == i]
        chain = chains[i] if chains and i < len(chains) else None
        gain = gains[i] if gains and i < len(gains) else 1.0

        if progress:
            progress(f"baking {os.path.basename(path)}")
        audio = render_track(path, [(0.0, duration)], track_mutes, chain, gain,
                             progress=progress, should_cancel=should_cancel)

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
        result = subprocess.run(cmd, capture_output=True,
                                creationflags=getattr(subprocess,
                                                      "CREATE_NO_WINDOW", 0))
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
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
    return float(result.stdout.strip())
