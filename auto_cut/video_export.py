"""
Renders a finished video file, for people who do not use DaVinci Resolve.

The FCPXML export hands Resolve a timeline and lets it do the work. This does
the work itself: keeps only the ranges that survived the edit, and muxes them
against the audio Wavefield already rendered - cuts, mutes, effects and levels
included.

The video HAS to be re-encoded. Cuts land wherever the speech stops, which is
almost never on a keyframe, and a stream copy can only cut on keyframes - it
would drift by up to several seconds per cut.

Each segment is encoded as its own short clip, then stitched together with
ffmpeg's concat demuxer (2026-09-13, replacing a single shared filter_complex
graph). The old design opened every camera source as a simultaneous input in
one ffmpeg process; with a multi-camera edit switching between sources across
hundreds of segments, `concat`'s strict requirement to emit segments in
original edit order meant whichever source's next needed segment was further
away kept decoding ahead of what `concat` could consume - those frames had
nowhere to go but memory. Measured on a 323-segment/3-camera episode: RAM
climbed toward full and CPU sat at a steady ~40% (buffering, not computing),
regardless of an earlier fix that removed redundant per-segment scale/pad
work. Encoding one source at a time, one segment at a time, removes the
cross-source interleaving entirely - there is nothing left to buffer ahead
of. It also stops decoding footage that is never used: each segment's own
ffmpeg call seeks directly to its own range instead of decoding its source
from the start.
"""

import os
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import bundled
from audio_export import _total_ram_bytes
from media_probe import probe

FFMPEG = bundled.tool("ffmpeg")
NL = chr(10)

# Constant Rate Factor. 20 is visually near-identical to a typical screen or
# webcam recording while roughly halving the size; lower is bigger and better.
# Only meaningful for NVENC (-cq) - the CPU fallback has no CRF-equivalent
# mode, see cpu_video_codec().
DEFAULT_CRF = 20

# Bits per pixel per frame for the CPU fallback's target bitrate - see
# cpu_video_codec(). Generous for the low-motion, talking-head webcam/screen
# content this app targets (same content DEFAULT_CRF's own comment is about),
# comfortably ahead of visible blocking without wildly oversizing the file.
_CPU_BITS_PER_PIXEL = 0.06

# Fast, keyframe-approximate seek distance placed before -i, in seconds. The
# residual (exact start minus this) is then a second -ss placed AFTER -i -
# sync_render.py established, by testing, that in this codebase that is what
# lands frame-accurately, not a plain before-i seek alone (sync_render.py:91).
# Long enough to guarantee landing before any keyframe near a segment's start
# even on a long-GOP source; short enough that decoding the residual stays
# cheap. Validate against real camera footage's actual GOP structure if cuts
# ever come back inaccurate - too short risks landing after the true
# keyframe.
VIDEO_SEEK_MARGIN = 5.0

# Below this, a machine is refused any parallel CPU segment encoding at all
# and stays on today's one-at-a-time path - it is already on the slower
# libopenh264 fallback (no working NVENC), which correlates with exactly
# the modest hardware this app's podcaster users often have, and is the
# machine least able to spare the extra memory concurrent ffmpeg processes
# would cost. Deliberately far below audio_export's own 15GB threshold:
# each parallel worker here only holds one short segment's decode/encode
# buffers, not a whole multi-hour track, so the risk is much smaller - this
# is a floor against genuinely constrained machines, not a scaled budget.
_CPU_ENCODE_MIN_RAM_BYTES = 8 * (1024 ** 3)


def _cpu_encode_workers():
    """
    How many libopenh264 (CPU fallback) segment encodes to run at once.

    Bound by CPU cores, leaving one free for the UI/OS, and capped low so a
    high-core-count machine doesn't spawn a pile of ffmpeg processes for
    diminishing returns. Unlike the GPU path there is no shared hardware
    encoder block to contend over, so this scales close to linearly with
    cores - this is the path that benefits podcasters without a high-end
    GPU the most, since they had zero segment parallelism before.
    """
    total_ram = _total_ram_bytes()
    if total_ram is not None and total_ram < _CPU_ENCODE_MIN_RAM_BYTES:
        return 1
    cores = os.cpu_count() or 2
    return max(1, min(cores - 1, 3))


def _gpu_encode_workers():
    """
    How many concurrent NVENC segment encodes to attempt.

    Deliberately not probed upfront the way has_nvenc() probes basic
    availability - testing *concurrent* NVENC would cost a real second
    encode, not two throwaway frames. Many budget/laptop GPUs (common
    hardware for this app's podcaster users) support only one concurrent
    NVENC session; some newer cards support several. Start at a
    conservative 2 and let _encode_segments' own retry-at-1 handle a GPU
    that turns out not to support this - see its docstring.
    """
    return 2


_nvenc_cache = None


def has_nvenc():
    """
    Whether NVIDIA hardware encoding actually WORKS here.

    Listing the encoders is not enough - it only says NVENC was compiled in.
    On this development machine ffmpeg lists h264_nvenc and then fails with
    "Driver does not support the required nvenc API version. Required: 13.1
    Found: 13.0", because the bundled ffmpeg is newer than the installed
    driver. The only reliable test is to encode something.

    So: two frames of black, to nowhere. Costs a fraction of a second, once.
    """
    global _nvenc_cache
    if _nvenc_cache is not None:
        return _nvenc_cache
    try:
        result = subprocess.run(
            [FFMPEG, "-hide_banner", "-f", "lavfi",
             "-i", "color=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        _nvenc_cache = result.returncode == 0
    except Exception:
        _nvenc_cache = False
    return _nvenc_cache


def cpu_video_codec(width, height, fps):
    """
    The CPU fallback encoder args, for when NVENC is unavailable.

    libopenh264, not libx264: this bundled ffmpeg build is deliberately
    LGPL-only (see packaging/THIRD-PARTY-NOTICES.txt - "This build contains
    no GPL-licensed components"), and ships no libx264 at all. Confirmed
    2026-09-13, when it turned out this whole CPU fallback path had never
    actually worked - `-c:v libx264` failed outright with "Unknown encoder",
    on every machine without a working NVENC. openh264 is already bundled,
    so this needed no relicensing or re-bundling to fix.

    openh264 has no CRF-equivalent quality mode, only bitrate-based rate
    control (`-rc_mode quality` still allocates bits adaptively within that
    budget, it just needs a budget) - see _CPU_BITS_PER_PIXEL.
    """
    bitrate = max(1_500_000, int(width * height * float(fps)
                                 * _CPU_BITS_PER_PIXEL))
    return ["-c:v", "libopenh264", "-rc_mode", "quality", "-b:v", str(bitrate)]


def _conform_vf(width, height, fps, src_width=None, src_height=None,
                fps_matches=False):
    """
    The scale/pad/fps/format/setpts chain every clip needs before concat, as
    a plain -vf string for one standalone ffmpeg call.

    Skips scale/pad when the clip's own source already matches width/height
    exactly - it was already a no-op there, just a wasted libswscale pass
    (measured contributing to a slow multi-camera export, 2026-09-13).

    Skips the fps filter when `fps_matches` - the source's own frame rate
    already exactly equals the target. This is not just an optimization:
    verified empirically (2026-09-13, via a frame-accuracy integration test)
    that applying `fps=` to an already-conformant, freshly-seeked stream
    shifts the selected frame back by one, regardless of its position
    relative to setpts - the fps filter's own frame-duplication/dropping
    grid does not align cleanly with a mid-stream seek. Skipping it when
    genuinely redundant avoids the bug entirely rather than working around
    its rounding.

    setsar/format/setpts always run - concat still needs every clip to
    agree on both, and setpts resets each clip's own timestamps to start at
    zero, which concat also depends on.
    """
    matches_size = src_width == width and src_height == height
    vf = "setpts=PTS-STARTPTS,setsar=1,format=yuv420p"
    if not fps_matches:
        vf = f"fps={fps}," + vf
    if matches_size:
        return vf
    # Letterbox rather than stretch: a camera framed differently should keep
    # its shape rather than be squashed to fit.
    return (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:-1:-1:color=black,") + vf


def _two_stage_seek(start):
    """
    (fast_seek, residual) for one segment starting at `start`.

    fast_seek goes before -i (cheap, keyframe-approximate); residual goes
    after -i as a second -ss (exact, decodes forward from the nearest
    keyframe to the real start) - see VIDEO_SEEK_MARGIN.
    """
    fast_seek = max(0.0, start - VIDEO_SEEK_MARGIN)
    return fast_seek, start - fast_seek


def _bookend_input(path, seconds, width, height, fps, use_gpu=False):
    """
    ffmpeg input arguments for one bookend, its kind, and (for a video
    bookend) its probed source dimensions and frame rate - used the same
    way as any other segment's source to decide whether _conform_vf can
    skip scale/pad and/or the fps filter.

    A video bookend contributes its own picture; audio-only gets black of
    the same length, already generated at exactly width x height x fps, so
    its dims/rate always trivially match the target (`src_fps=None` signals
    that to the caller, same as the width/height convention below).
    """
    if seconds <= 0:
        return [], None, None, None, None
    if path:
        try:
            info = probe(path)
            if info.has_video:
                # Trimmed to the audio length so picture and sound agree even
                # if the file is slightly longer.
                hwaccel = ["-hwaccel", "cuda"] if use_gpu else []
                return (hwaccel + ["-t", f"{seconds:.6f}", "-i", path],
                        "video", info.width, info.height, info.fps)
        except Exception:
            pass                         # unreadable: fall through to black
    return (["-f", "lavfi", "-t", f"{seconds:.6f}",
             "-i", f"color=black:s={width}x{height}:r={fps}"],
            "black", width, height, None)


def _discard(path):
    """Removes a scratch file without ever becoming the reason an export failed."""
    try:
        os.remove(path)
    except OSError:
        pass


def _discard_all(paths):
    for path in paths:
        _discard(path)


class _Cancelled(Exception):
    """Raised internally to unwind out of a render on should_cancel()."""


def _run_encode(cmd, out_path, on_time, should_cancel):
    """
    Runs one ffmpeg encode, calling `on_time(elapsed_seconds)` as ffmpeg
    reports its own `time=` position. Reports only THIS clip's own elapsed
    time, not a fraction of an overall total - clips used to complete in a
    fixed order, so a simple running `base_completed` offset was enough,
    but parallel segment encoding means completion order is no longer
    predictable. See `_ProgressAggregator`, which combines multiple clips'
    elapsed time into one overall fraction.

    Raises `_Cancelled` (with the partial file already removed) if
    `should_cancel` fires mid-encode, or `RuntimeError` with ffmpeg's stderr
    tail on a real failure.
    """
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", errors="replace",
                                   bufsize=1)
    except Exception:
        _discard(out_path)
        raise
    tail = []
    time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")
    try:
        for line in process.stderr:
            tail.append(line)
            del tail[:-40]              # keep only enough to explain a failure
            if should_cancel and should_cancel():
                process.terminate()
                process.wait(timeout=10)
                _discard(out_path)
                raise _Cancelled()
            match = time_pattern.search(line)
            if match and on_time:
                hours, minutes, seconds = match.groups()
                elapsed = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                on_time(elapsed)
        process.wait()
    finally:
        if process.poll() is None:
            process.kill()
    if process.returncode != 0:
        _discard(out_path)
        raise RuntimeError("".join(tail))


class _ProgressAggregator:
    """
    Combines every clip's own ffmpeg `time=` elapsed-seconds report into one
    overall fraction against a fixed `total`. Needed once segments can
    encode in parallel: the old design could just add each clip's elapsed
    time to a running `base_completed`, because clips always finished (and
    were reported) in a fixed order. Thread-safe - clips running in
    different pool workers report concurrently.
    """

    def __init__(self, total, progress, message):
        self._total = total
        self._progress = progress
        self._message = message
        self._elapsed = {}
        self._lock = threading.Lock()

    def track(self, clip_id):
        def on_time(elapsed):
            if not self._progress:
                return
            with self._lock:
                self._elapsed[clip_id] = elapsed
                done = sum(self._elapsed.values())
            if self._total > 0:
                self._progress(min(1.0, done / self._total), self._message)
        return on_time

    def finish(self, clip_id, duration):
        # Pins this clip's contribution to its real duration once it's done,
        # rather than whatever its last `time=` line happened to report -
        # ffmpeg's own progress lines can lag slightly behind completion.
        with self._lock:
            self._elapsed[clip_id] = duration


def _encode_segments(clip_specs, video_codec, use_gpu, aggregator,
                     should_cancel):
    """
    Runs every segment's ffmpeg encode through a small bounded worker pool
    instead of one at a time. Segments share no state across each other
    (unlike an audio VST chain's continuous stream across a whole track),
    so nothing but wall-clock time was ever gained by serializing them.
    `clip_specs` is `[(clip_id, clip_path, input_args, vf, duration)]`.

    Worker count comes from `_cpu_encode_workers`/`_gpu_encode_workers`,
    chosen conservatively for the kind of modest hardware this app's
    podcaster users often have.

    A NVENC session limit is a real risk on budget/laptop GPUs (some
    support only one concurrent encode) and isn't something worth probing
    upfront the way `has_nvenc()` probes basic availability - testing
    *concurrent* NVENC would cost a real second encode, not two throwaway
    frames. So instead: if running at concurrency >1 on the GPU path fails
    at all, retry the whole batch once at concurrency 1 before giving up.
    If it still fails, the error is left to `render()`'s own outer handler,
    which already restarts the whole video phase on CPU for a genuine GPU
    failure - this function doesn't duplicate that decision.

    `should_cancel()` firing, or any segment failing for a reason that
    isn't a GPU-concurrency retry, cancels every other in-flight and
    pending segment and raises - matching this module's existing
    fail-fast, clean-up-and-abort behavior.
    """
    workers = _gpu_encode_workers() if use_gpu else _cpu_encode_workers()
    cancel_event = threading.Event()

    def combined_cancel():
        return cancel_event.is_set() or bool(should_cancel and should_cancel())

    def run_one(clip_id, clip_path, input_args, vf, duration):
        if combined_cancel():
            raise _Cancelled()
        cmd = _clip_cmd(input_args, vf, video_codec, clip_path)
        _run_encode(cmd, clip_path, aggregator.track(clip_id), combined_cancel)
        aggregator.finish(clip_id, duration)

    def run_batch(specs, batch_workers):
        with ThreadPoolExecutor(max_workers=max(1, batch_workers)) as pool:
            futures = [pool.submit(run_one, *spec) for spec in specs]
            error = None
            for future in as_completed(futures):
                exc = future.exception()
                if exc is not None and error is None:
                    error = exc
                    cancel_event.set()
            if error is not None:
                raise error

    if not use_gpu or workers <= 1:
        run_batch(clip_specs, workers)
        return

    try:
        run_batch(clip_specs, workers)
    except _Cancelled:
        raise
    except Exception:
        cancel_event.clear()
        run_batch(clip_specs, 1)


def _clip_cmd(input_args, vf, video_codec, out_path):
    return [
        FFMPEG, "-y", "-hide_banner",
        *input_args,
        "-vf", vf,
        "-an",                          # picture only - audio is muxed once, at the end
        *video_codec,
        "-pix_fmt", "yuv420p",
        # Every clip must agree on timestamp behavior or the final concat
        # -c copy can introduce gaps/drift at splice points even when the
        # codec/resolution/pix_fmt all already match.
        "-avoid_negative_ts", "make_zero",
        "-video_track_timescale", "90000",
        out_path,
    ]


def render(video_path, audio_path, out_path, keep_ranges, crf=DEFAULT_CRF,
           use_gpu=None, progress=None, should_cancel=None,
           intro_seconds=0.0, outro_seconds=0.0,
           intro_path=None, outro_path=None,
           segments=None, sources=None):
    """
    Writes `out_path` from `video_path`'s picture and `audio_path`'s sound.

    For a vodcast, pass `sources` (the camera files, V1/V2/V3) and `segments`
    ([(camera_index, start, end)] from scenes.apply_to_keep_ranges); the cut
    then moves between cameras. Without them every segment comes from
    `video_path` and this is an ordinary single-camera export.

    V3 is only ever a picture source - its audio is the same two voices again
    and is never opened.

    `audio_path` is expected to already include any intro and outro; pass their
    durations so matching black can be put in front of and after the picture.

    progress(fraction, message) is called as ffmpeg reports its position.
    should_cancel() is polled; returning True stops the render and removes the
    partial file.
    """
    if segments is None:
        segments = [(0, start, end) for start, end in keep_ranges]
    if not segments:
        raise ValueError("Nothing to export - no segments survived the edit.")
    if not sources:
        sources = [video_path]

    # The first camera sets the format everything else is conformed to.
    info = probe(sources[0])
    # Probed once per source so _conform_vf can skip scale/pad for whichever
    # segments already match.
    source_infos = [info] + [probe(s) for s in sources[1:]]
    total = (sum(end - start for _source, start, end in segments)
             + intro_seconds + outro_seconds)
    if use_gpu is None:
        use_gpu = has_nvenc()

    # NVENC needs a rate-control mode named explicitly; -cq on its own is
    # rejected with "Invalid argument" and no useful explanation.
    gpu_codec = ["-c:v", "h264_nvenc", "-preset", "p4",
                 "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    cpu_codec = cpu_video_codec(info.width, info.height, info.fps)
    video_codec = gpu_codec if use_gpu else cpu_codec

    rate = f"{float(info.fps):.6f}"
    width, height = info.width, info.height

    if progress:
        progress(0.0, f"encoding with {'GPU' if use_gpu else 'CPU'} "
                      f"({total / 60:.1f} min of video)")

    temp_dir = tempfile.mkdtemp(prefix="wavefield_video_")
    clip_paths = []
    try:
        aggregator = _ProgressAggregator(total, progress, "Encoding video...")

        intro_args, intro_kind, intro_w, intro_h, intro_fps = _bookend_input(
            intro_path, intro_seconds, width, height, rate, use_gpu=use_gpu)
        outro_args, outro_kind, outro_w, outro_h, outro_fps = _bookend_input(
            outro_path, outro_seconds, width, height, rate, use_gpu=use_gpu)
        if progress and (intro_kind == "video" or outro_kind == "video"):
            progress(0.0, "using the picture from your intro/outro")

        if intro_args:
            clip = os.path.join(temp_dir, "clip_intro.mp4")
            vf = _conform_vf(width, height, rate, intro_w, intro_h,
                             fps_matches=(intro_fps is None
                                          or intro_fps == info.fps))
            _run_encode(_clip_cmd(intro_args, vf, video_codec, clip), clip,
                       aggregator.track("intro"), should_cancel)
            aggregator.finish("intro", intro_seconds)
            clip_paths.append(clip)

        # Every segment's own args are built upfront, in original edit order,
        # and appended to clip_paths here - not as encoding finishes. Concat
        # order only depends on this list's order, so it stays correct
        # regardless of which segment's encode actually finishes first once
        # they run in parallel below.
        clip_specs = []
        for i, (source, start, end) in enumerate(segments):
            clip = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
            duration = end - start
            src_info = (source_infos[source]
                       if 0 <= source < len(source_infos) else None)
            vf = _conform_vf(width, height, rate,
                             getattr(src_info, "width", None),
                             getattr(src_info, "height", None),
                             fps_matches=(src_info is not None
                                          and src_info.fps == info.fps))
            fast_seek, residual = _two_stage_seek(start)
            input_args = []
            if use_gpu:
                input_args += ["-hwaccel", "cuda"]
            input_args += ["-ss", f"{fast_seek:.6f}", "-i", sources[source],
                           "-ss", f"{residual:.6f}", "-t", f"{duration:.6f}"]
            clip_specs.append((i, clip, input_args, vf, duration))
            clip_paths.append(clip)

        _encode_segments(clip_specs, video_codec, use_gpu, aggregator,
                         should_cancel)

        if outro_args:
            clip = os.path.join(temp_dir, "clip_outro.mp4")
            vf = _conform_vf(width, height, rate, outro_w, outro_h,
                             fps_matches=(outro_fps is None
                                          or outro_fps == info.fps))
            _run_encode(_clip_cmd(outro_args, vf, video_codec, clip), clip,
                       aggregator.track("outro"), should_cancel)
            aggregator.finish("outro", outro_seconds)
            clip_paths.append(clip)

        if should_cancel and should_cancel():
            raise _Cancelled()

        if progress:
            progress(0.98, "joining clips...")
        concat_list = os.path.join(temp_dir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for path in clip_paths:
                escaped = path.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        video_only = os.path.join(temp_dir, "video_only.mp4")
        result = subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c", "copy", video_only],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        if should_cancel and should_cancel():
            raise _Cancelled()

        if progress:
            progress(0.99, "adding audio...")
        result = subprocess.run(
            [FFMPEG, "-y", "-hide_banner",
             "-i", video_only, "-i", audio_path,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", out_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    except _Cancelled:
        if os.path.exists(out_path):
            _discard(out_path)
        return None
    except RuntimeError as exc:
        message = str(exc)
        lowered = message.lower()
        # A GPU encode can still fail after passing the probe - a driver
        # update mid-session, another program holding the encoder, or the
        # -hwaccel cuda decode failing for a source the GPU decoder can't
        # handle. The CPU path is slower but always works, and is far better
        # than handing someone an ffmpeg backtrace. Restarts the whole video
        # phase on CPU rather than mixing NVENC- and libx264-encoded clips in
        # one concat, which is untested and a real corruption/desync risk.
        if use_gpu and ("nvenc" in lowered or "cuda" in lowered
                        or "hwaccel" in lowered):
            if progress:
                progress(0.0, "GPU encoder unavailable - encoding on the "
                              "processor instead (slower)")
            return render(video_path, audio_path, out_path, keep_ranges,
                          crf=crf, use_gpu=False, progress=progress,
                          should_cancel=should_cancel,
                          intro_seconds=intro_seconds,
                          outro_seconds=outro_seconds,
                          intro_path=intro_path, outro_path=outro_path,
                          segments=segments, sources=sources)
        raise RuntimeError("ffmpeg could not write the video:" + NL + NL
                           + NL.join(message.splitlines()[-12:]))
    finally:
        _discard_all(clip_paths)
        _discard(os.path.join(temp_dir, "concat.txt"))
        _discard(os.path.join(temp_dir, "video_only.mp4"))
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    return out_path
