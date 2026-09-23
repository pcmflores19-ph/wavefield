"""
Turns a sync.SyncResult's offset into an actual file.

Never touches the original media - always writes a new, cached copy and
lets the caller point speaker_paths at that instead. A trim re-encodes
(an accurate leading cut can't land on a keyframe, so `-c copy` would leave
a GOP's worth of untrimmed material); a pad prepends silence, and black
video if the source has a picture, via a single filter_complex pass.
"""

import hashlib
import os
import subprocess

import bundled
import settings
from media_probe import probe
from video_export import detect_gpu_encoder, gpu_video_codec, cpu_video_codec

FFMPEG = bundled.tool("ffmpeg")

# Higher quality than the 20 used for the final export pass - this derived
# file becomes the new working source for everything downstream (analysis,
# further edits, eventual export), not a one-time deliverable, so it's worth
# spending more bits to avoid compounding quality loss.
_CRF = 18


def _video_codec(width, height, fps):
    encoder = detect_gpu_encoder()
    if encoder:
        return gpu_video_codec(encoder, _CRF)
    # libopenh264, not libx264 - see video_export.cpu_video_codec's
    # docstring: this bundled ffmpeg build has no libx264 at all.
    return cpu_video_codec(width, height, fps)


def _synced_cache_path(path, offset_seconds, has_video):
    """
    Flat, not a subdirectory - settings._cache_entries() only lists files
    directly under cache_dir() (os.listdir, not recursive), so a synced/
    subfolder's contents would be invisible to cache size/prune/clear.
    Keyed on the offset too, not just the source file: a different
    reference-track choice or a re-run must never collide with a stale copy.

    Extension is chosen by content, NOT copied from the source file - the
    source's own container often can't legally hold what we're about to
    write into it (confirmed: syncing an .mp3 crashed ffmpeg outright,
    "Exactly one MP3 audio stream is required", because the mp3 muxer
    can't hold pcm_s16le at all - it needs actual MP3-encoded audio, and
    plenty of other extensions - .m4a, .aac, .ogg - are just as strict).
    .mov for video, same reasoning bake_processed_media already uses for
    exactly this ("audio is copied in as 16-bit PCM... never re-encoded");
    .wav for audio-only, the plain PCM container everything else in this
    app already treats as the safe intermediate format.
    """
    stat = os.stat(path)
    key = hashlib.sha1(
        f"{path}|{stat.st_size}|{stat.st_mtime}|sync2|{offset_seconds:.4f}"
        .encode("utf-8")
    ).hexdigest()
    directory = settings.cache_dir()
    os.makedirs(directory, exist_ok=True)
    ext = ".mov" if has_video else ".wav"
    return os.path.join(directory, f"sync_{key}{ext}")


def render_synced_copy(path, offset_seconds, log=None):
    """
    Writes (or reuses a cached) copy of `path` with `offset_seconds` applied
    - positive trims that much off the front, negative pads that much
    silence (and black video, if there's a picture stream) onto it. Returns
    the copy's path. Raises RuntimeError with ffmpeg's stderr on failure.
    """
    info = probe(path)
    has_video = info.has_video
    out_path = _synced_cache_path(path, offset_seconds, has_video)
    if os.path.exists(out_path):
        if log:
            log(f"  using cached sync copy of {os.path.basename(path)}")
        return out_path

    video_codec = (_video_codec(info.width, info.height, info.fps)
                  if has_video else [])

    # pcm_s16le, not aac/copy: this derived file is the new working source
    # for everything downstream (same reasoning bake_processed_media uses
    # for keeping its own audio lossless) - re-encoding video is
    # unavoidable for an accurate trim/pad, but audio doesn't have to lose
    # anything in the process. Always re-encoded into .mov/.wav (see
    # _synced_cache_path), never the source's own container/codec.
    audio_codec = ["-c:a", "pcm_s16le"]

    if offset_seconds > 0:
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-ss", f"{offset_seconds:.6f}",     # after -i: accurate, not keyframe-snapped
            "-i", path,
            *video_codec, *audio_codec,
            out_path,
        ]
    elif offset_seconds < 0:
        pad = -offset_seconds
        # all=1: apply the delay to every channel regardless of count,
        # rather than the default first-two-channels-only syntax.
        delay_ms = int(round(pad * 1000))
        if has_video:
            filt = (
                f"[0:v]tpad=start_duration={pad:.6f}:color=black[v];"
                f"[0:a]adelay={delay_ms}:all=1[a]"
            )
            maps = ["-map", "[v]", "-map", "[a]"]
            codec = video_codec
        else:
            filt = f"[0:a]adelay={delay_ms}:all=1[a]"
            maps = ["-map", "[a]"]
            codec = []
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-i", path,
            "-filter_complex", filt,
            *maps, *codec, *audio_codec,
            out_path,
        ]
    else:
        # Not called by the normal sync worker (it skips zero-offset tracks
        # entirely), but kept correct for direct callers: -c copy would
        # assume the source's own codec is legal in .mov/.wav, which is
        # exactly the assumption that just broke for .mp3.
        cmd = [
            FFMPEG, "-y", "-v", "error", "-i", path,
            *video_codec, *audio_codec,
            out_path,
        ]

    if log:
        log(f"  rendering synced copy of {os.path.basename(path)} "
            f"({offset_seconds:+.3f}s) ...")
    result = subprocess.run(cmd, capture_output=True,
                            creationflags=getattr(subprocess,
                                                  "CREATE_NO_WINDOW", 0))
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not render a synced copy of {os.path.basename(path)}:\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return out_path
