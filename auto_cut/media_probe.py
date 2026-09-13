"""
Reads frame rate, resolution and duration out of a media file via ffprobe,
so the generated FCPXML matches the real source media.
"""

import json
import subprocess
from fractions import Fraction

import bundled

FFPROBE = bundled.tool("ffprobe")


def _timecode_frames(timecode, fps):
    """
    "01:00:00:00" at 24fps -> 86400 frames. Returns 0 for anything unparseable,
    which is the right answer for media that carries no timecode.

    A drop-frame separator (";") is read as non-drop: the error is 0.1% and only
    affects where a clip's in-point sits inside the source, never the cut
    timings on the timeline.
    """
    if not timecode:
        return 0
    parts = timecode.replace(";", ":").split(":")
    if len(parts) != 4:
        return 0
    try:
        hours, minutes, seconds, frames = (int(p) for p in parts)
    except ValueError:
        return 0
    rate = int(round(float(fps)))
    return ((hours * 3600 + minutes * 60 + seconds) * rate) + frames


class MediaInfo:
    def __init__(self, path, fps, width, height, duration_seconds, has_audio,
                 audio_channels=1, audio_rate=48000, start_timecode="00:00:00:00"):
        self.path = path
        self.fps = fps  # Fraction, e.g. Fraction(30000, 1001)
        self.width = width
        self.height = height
        self.duration_seconds = duration_seconds
        self.has_audio = has_audio
        self.audio_channels = audio_channels
        self.audio_rate = audio_rate
        # Cameras habitually stamp 01:00:00:00. The media's frames live at that
        # timecode, so a timeline asking for them at 0s asks for frames the file
        # does not contain - Resolve finds the file and reports Media Offline.
        self.start_timecode = start_timecode

    @property
    def start_frames(self):
        """The media's start timecode as a whole number of its own frames."""
        return _timecode_frames(self.start_timecode, self.fps)

    @property
    def has_video(self):
        return self.width > 0 and self.height > 0

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
    audio_rate = int(audio.get("sample_rate", 48000)) if audio else 48000

    # ffprobe reports the timecode track as a format tag, or as a stream tag on
    # the video or the timecode stream itself, depending on the container.
    start_timecode = data.get("format", {}).get("tags", {}).get("timecode")
    if not start_timecode:
        for stream in streams:
            tag = stream.get("tags", {}).get("timecode")
            if tag:
                start_timecode = tag
                break

    # A plain WAV or MP3 is a perfectly good source for an audio podcast, so
    # audio-only files are allowed. They just cannot be exported as a timeline
    # or a video - the app disables those when no source has a picture.
    if video is None:
        duration = audio.get("duration") or data.get("format", {}).get("duration")
        if duration is None:
            raise RuntimeError(f"Could not determine duration for {path}")
        return MediaInfo(path, Fraction(30, 1), 0, 0, float(duration),
                         has_audio, audio_channels=audio_channels,
                         audio_rate=audio_rate,
                         start_timecode=start_timecode or "00:00:00:00")

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
                     audio_channels=audio_channels, audio_rate=audio_rate,
                     start_timecode=start_timecode or "00:00:00:00")
