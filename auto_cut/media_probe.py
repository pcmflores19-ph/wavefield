"""
Reads frame rate, resolution and duration out of a media file via ffprobe,
so the generated FCPXML matches the real source media.
"""

import json
import subprocess
from fractions import Fraction

import bundled

FFPROBE = bundled.tool("ffprobe")


class MediaInfo:
    def __init__(self, path, fps, width, height, duration_seconds, has_audio,
                 audio_channels=1):
        self.path = path
        self.fps = fps  # Fraction, e.g. Fraction(30000, 1001)
        self.width = width
        self.height = height
        self.duration_seconds = duration_seconds
        self.has_audio = has_audio
        self.audio_channels = audio_channels

    @property
    def has_video(self):
        return self.width > 0 and self.height > 0

    @property
    def fps_float(self):
        return float(self.fps)

    def __repr__(self):
        return (f"MediaInfo({self.path!r}, {self.fps} fps, {self.width}x{self.height}, "
                f"{self.duration_seconds:.2f}s, audio={self.has_audio})")


def probe(path):
    cmd = [
        FFPROBE, "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None and audio is None:
        raise RuntimeError(f"No audio or video stream found in {path}")
    has_audio = audio is not None
    # Declared in the FCPXML: telling Resolve a mono source is stereo made it
    # allocate extra audio tracks on import.
    audio_channels = int(audio.get("channels", 1)) if audio else 1

    # A plain WAV or MP3 is a perfectly good source for an audio podcast, so
    # audio-only files are allowed. They just cannot be exported as a timeline
    # or a video - the app disables those when no source has a picture.
    if video is None:
        duration = audio.get("duration") or data.get("format", {}).get("duration")
        if duration is None:
            raise RuntimeError(f"Could not determine duration for {path}")
        return MediaInfo(path, Fraction(30, 1), 0, 0, float(duration),
                         has_audio, audio_channels=audio_channels)

    # r_frame_rate is the real (not average) rate, e.g. "30000/1001"
    fps = None
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = video.get(key)
        if raw and raw != "0/0":
            try:
                candidate = Fraction(raw)
                if candidate > 0:
                    fps = candidate
                    break
            except (ZeroDivisionError, ValueError):
                pass
    if fps is None or fps <= 0:
        fps = Fraction(30, 1)

    width = int(video.get("width", 1920))
    height = int(video.get("height", 1080))

    duration = video.get("duration") or data.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"Could not determine duration for {path}")

    return MediaInfo(path, fps, width, height, float(duration), has_audio,
                     audio_channels=audio_channels)
