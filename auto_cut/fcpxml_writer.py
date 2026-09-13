"""
Writes an FCPXML timeline containing only the "keep" ranges, for importing
into DaVinci Resolve (File > Import > Timeline > ...).

The first speaker goes in the primary storyline and lands on V1/A1. Every
other speaker is a connected clip on lane N, landing on V(N+1)/A(N+1). Two
speakers therefore import as exactly two video and two audio tracks.

An earlier version hung all speakers off a base <gap> instead, because putting
a speaker in the primary storyline had made Resolve splinter that speaker's
audio across extra tracks. That gap then occupied the primary storyline and
showed up as an empty V1. The scattering turned out to have two causes, both
since fixed: the assets declared audioChannels="2" for what are mono
recordings, and every speaker shared one generic "dialogue" role, so Resolve
could not tell whose audio was whose once the timeline fragmented into many
cut segments. With honest mono and a unique role per speaker, the storyline
structure imports cleanly and there is no empty track.

Host and guest are split on the UNION of both their mute boundaries, so their
pieces line up one to one and each guest piece can hang off the host piece it
sits over. A connected clip's offset is measured in its parent clip's local
time, which is why those offsets equal the source start rather than the
position on the timeline.

FCPXML expresses time as exact rationals (e.g. "1001/30000s"), so everything
here is computed in whole frames to avoid drift.
"""

import os
from fractions import Fraction
from xml.sax.saxutils import escape

FCPXML_VERSION = "1.8"

NL = chr(10)


def _time_str(frames, fps):
    """
    Rational time string for a whole number of frames at the given fps.
    fps is a Fraction, e.g. Fraction(30000, 1001) -> N frames is
    N * 1001/30000 seconds.
    """
    if frames == 0:
        return "0s"
    numerator = frames * fps.denominator
    denominator = fps.numerator
    common = Fraction(numerator, denominator)
    if common.denominator == 1:
        return f"{common.numerator}s"
    return f"{common.numerator}/{common.denominator}s"


def _frame_duration_str(fps):
    return _time_str(1, fps)


def _file_url(path):
    abs_path = os.path.abspath(path).replace("\\", "/")
    if not abs_path.startswith("/"):
        abs_path = "/" + abs_path
    from urllib.parse import quote
    # Keep the Windows drive colon literal (file:///C:/...) - Resolve fails to
    # relink media if it's percent-encoded as C%3A.
    return "file://" + quote(abs_path, safe="/:")


# <sequence audioRate> is an enumeration in the DTD, not a number:
# <!ENTITY % audioHz "( 32k | 44.1k | 48k | 88.2k | 96k | 176.4k | 192k )">
# An unlisted rate has to be reported as the nearest legal one - writing the
# true value would make the document invalid rather than merely imprecise.
_AUDIO_RATES = (32000, 44100, 48000, 88200, 96000, 176400, 192000)


def _audio_rate_label(rate):
    """48000 -> "48k", 44100 -> "44.1k"."""
    nearest = min(_AUDIO_RATES, key=lambda candidate: abs(candidate - rate))
    return f"{nearest / 1000:g}k"


def _format_name(width, height, fps):
    fps_label = round(float(fps), 2)
    if float(fps).is_integer():
        fps_label = int(float(fps))
    return f"FFVideoFormat{height}p{fps_label}"


def _speaker_roles(speaker_media):
    """
    Assigns each speaker a distinct FCPXML audio subrole (e.g. "dialogue.host",
    "dialogue.guest2") derived from their filename. Resolve groups clips onto
    audio tracks by role - giving every speaker's clips the same generic
    "dialogue" role (as v1 did) left Resolve unable to tell whose audio was
    whose once the timeline fragmented into many cut segments, so it scattered
    them across tracks unpredictably. A unique role per speaker keeps each
    speaker on their own dedicated, contiguous audio track.
    """
    used = set()
    roles = []
    for media in speaker_media:
        base = os.path.splitext(os.path.basename(media.path))[0].lower()
        token = "".join(c for c in base if c.isalnum()) or "speaker"
        role = token
        n = 2
        while role in used:
            role = f"{token}{n}"
            n += 1
        used.add(role)
        roles.append(f"dialogue.{role}")
    return roles


def _aligned_pieces(seg_start_f, seg_end_f, mutes_by_speaker, speaker_count,
                    extra_cut_frames=None):
    """
    Splits a source frame range on EVERY speaker's mute boundaries at once,
    returning [(from_frame, to_frame, [muted_per_speaker]), ...].

    Splitting each speaker independently would give them different boundaries,
    and a connected clip has to hang off the storyline clip it overlaps - so
    the pieces have to line up.

    `extra_cut_frames`, if given, adds more cut points (e.g. scene-switch
    boundaries) to the same union - purely additive, so a caller that never
    passes it sees identical behavior to before this parameter existed.
    """
    cuts = set()
    for index in range(speaker_count):
        for m_start, m_end in mutes_by_speaker.get(index, []):
            if m_end > seg_start_f and m_start < seg_end_f:
                cuts.add(max(m_start, seg_start_f))
                cuts.add(min(m_end, seg_end_f))
    if extra_cut_frames:
        for f in extra_cut_frames:
            if seg_start_f < f < seg_end_f:
                cuts.add(f)
    bounds = sorted({seg_start_f, seg_end_f} | cuts)

    pieces = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2
        muted = [any(m_start <= mid < m_end
                     for m_start, m_end in mutes_by_speaker.get(i, []))
                 for i in range(speaker_count)]
        pieces.append((a, b, muted))
    return pieces


def build_fcpxml(speaker_media, keep_ranges, project_name="Podcast (Wavefield)",
                 mutes=None, scenes=None, camera_media=None):
    """
    speaker_media: list of MediaInfo (from media_probe.probe), speaker 1 first.
                   Their timelines are assumed to start together at 0.
    keep_ranges:   list of (start_seconds, end_seconds) in timeline time.
    mutes:         optional [(speaker_index, start_seconds, end_seconds)] -
                   those stretches are emitted as silenced sub-clips.
    scenes:        optional [(camera_index, start_seconds, end_seconds)] -
                   scenes_mod.apply_to_keep_ranges's output, covering the kept
                   timeline with no gaps. Requires camera_media too.
    camera_media:  optional [MediaInfo, MediaInfo, MediaInfo] for V1/V2/V3 -
                   V1/V2 should be the SAME objects as speaker_media[0]/[1]
                   (same source file; a SEPARATE, video-only asset is still
                   declared for their picture lanes - see picture_asset_ids
                   below for why).

    When scenes/camera_media are given, three extra always-present connected
    clip lanes are added above the existing per-speaker lanes - one per
    camera, each permanently referencing that one camera's own dedicated,
    audio-free asset for the whole file, toggled visible ("enabled") only
    for the piece(s) where scenes says that camera is active (each camera's
    own audio is already carried by the spine/speaker lanes above, so the
    picture lane needs none). Keeping each lane's asset reference fixed and
    only toggling `enabled` is deliberately different from switching WHICH
    asset a lane points to piece to piece - the latter is what made Resolve
    rearrange clips in an earlier attempt (see scenes.py's module docstring).

    Without both scenes and camera_media (the default, and every pre-existing
    caller), output is unchanged from before this parameter existed.
    Returns the FCPXML document as a string.
    """
    if not speaker_media:
        raise ValueError("Need at least one speaker's media to build a timeline.")
    if bool(scenes) != bool(camera_media):
        raise ValueError("scenes and camera_media must be given together.")

    # The sequence's frame rate follows the first speaker, but its
    # resolution is always 1080p regardless of the source recordings' own
    # resolution - this is the timeline CANVAS (format "r0"), never an
    # asset's own attributes (assets carry no width/height at all in this
    # file, only a "format" reference), so forcing it here doesn't affect
    # source relinking.
    base = speaker_media[0]
    fps = base.fps
    frame_dur = _frame_duration_str(fps)
    SEQUENCE_WIDTH, SEQUENCE_HEIGHT = 1920, 1080

    resources = []
    resources.append(
        f'    <format id="r0" name="{_format_name(SEQUENCE_WIDTH, SEQUENCE_HEIGHT, fps)}" '
        f'frameDuration="{frame_dur}" width="{SEQUENCE_WIDTH}" height="{SEQUENCE_HEIGHT}" '
        f'colorSpace="1-1-1 (Rec. 709)"/>'
    )

    # Where each speaker's media sits on its own timecode. Cameras stamp
    # 01:00:00:00, and an asset claiming to start at 0s while its frames live an
    # hour in makes Resolve import the clip as Media Offline: it finds the file,
    # then finds nothing at the times the timeline asks for. Every source
    # in-point below is measured from the media's real start, not from zero.
    #
    # Two sets of numbers, because a clip's attributes are in the SEQUENCE's
    # timebase while an asset's are in its own: media_tc_frames counts the
    # media's own frames, tc_frames the equivalent count of sequence frames.
    # They only differ when a camera ran at a different rate.
    media_tc_frames = {media.path: getattr(media, "start_frames", 0)
                       for media in speaker_media}
    tc_frames = {
        media.path: int(round(media_tc_frames[media.path]
                              / float(media.fps) * float(fps)))
        for media in speaker_media
    }
    base_tc = tc_frames[speaker_media[0].path]

    def asset_resource_xml(asset_id, media, include_audio=True):
        # An asset's own start and duration are in ITS OWN timebase, which is
        # not the sequence's when speakers were shot on cameras running at
        # different rates.
        own_fps = media.fps
        total_frames = int(round(media.duration_seconds * float(own_fps)))
        name = escape(os.path.splitext(os.path.basename(media.path))[0])
        video_attrs = ' hasVideo="1" format="r0"' if media.has_video else ""
        audio_attrs = ""
        if include_audio and media.has_audio:
            audio_attrs = (
                f' hasAudio="1" audioSources="1"'
                f' audioChannels="{getattr(media, "audio_channels", 1) or 1}"'
                f' audioRate="{getattr(media, "audio_rate", 48000)}"'
            )
        # Both forms of the file reference, deliberately: the DTD declares
        # <!ATTLIST asset src CDATA #REQUIRED>, while Resolve only actually
        # relinks from the <media-rep> child - given src alone it asks which
        # folder the media is in and then finds no clip there.
        file_url = _file_url(media.path)
        return (
            f'    <asset id="{asset_id}" name="{name}" src="{file_url}" '
            f'start="{_time_str(media_tc_frames[media.path], own_fps)}" '
            f'duration="{_time_str(total_frames, own_fps)}"'
            f'{video_attrs}{audio_attrs}>' + NL +
            f'      <media-rep kind="original-media" src="{file_url}"/>' + NL +
            f'    </asset>'
        )

    asset_ids = {}
    for i, media in enumerate(speaker_media, start=1):
        asset_id = f"r{i}"
        asset_ids[media.path] = asset_id
        resources.append(asset_resource_xml(asset_id, media))

    # Every picture lane gets its OWN video-only asset, declared with no
    # audio at all - even for V1/V2, which already have a full audio-
    # carrying asset above (r1/r2, used by the spine/speaker lanes).
    #
    # Tried first: referencing that SAME r1/r2 asset from the picture lane
    # and suppressing its audio at the clip level (an <adjust-volume -96dB>
    # gain, then srcEnable="video"). Both left Resolve creating a full extra
    # (merely silent) audio track per picture-lane clip on direct import
    # 2026-09-13 - confirmed by comparing against V3's picture lane, which
    # never got a duplicate track precisely because ITS asset never declares
    # audio in the first place. Audio-track creation follows the ASSET's own
    # declared capability, not anything at the clip-instance level - so
    # every picture lane, including V1/V2's, needs a dedicated no-audio
    # asset the same way V3 already gets one below.
    picture_asset_ids = {}
    if camera_media:
        for media in camera_media:
            asset_id = f"r{len(asset_ids) + len(picture_asset_ids) + 1}"
            picture_asset_ids[media.path] = asset_id
            media_tc_frames.setdefault(media.path, getattr(media, "start_frames", 0))
            tc_frames.setdefault(media.path, int(round(
                media_tc_frames[media.path] / float(media.fps) * float(fps))))
            resources.append(asset_resource_xml(asset_id, media, include_audio=False))

    roles = _speaker_roles(speaker_media)

    # Muted spans, per speaker, in source frames.
    mute_frames_by_speaker = {}
    for speaker_index, m_start, m_end in (mutes or []):
        mute_frames_by_speaker.setdefault(speaker_index, []).append(
            (int(round(m_start * float(fps))), int(round(m_end * float(fps))))
        )

    # Convert keep ranges to whole frames and lay them end to end on the
    # timeline; each keeps its own source in-point. Speaker 0 forms the primary
    # storyline, everyone else hangs off it on a lane (see module docstring).
    spine_entries = []
    timeline_cursor_frames = 0

    def clip_xml(indent, media, lane, offset_f, start_f, dur_f, muted,
                 role, children="", enabled=True, video_only=False, asset_id=None):
        name = escape(os.path.splitext(os.path.basename(media.path))[0])
        lane_attr = f'lane="{lane}" ' if lane else ""
        role_attr = f' audioRole="{role}"' if role else ""
        # Omitted rather than written as "1": FCPXML treats a clip as enabled
        # by default, and every pre-existing call site relies on that same
        # emitted text - only a picture-lane clip explicitly toggled off ever
        # passes enabled=False.
        enabled_attr = "" if enabled else ' enabled="0"'
        # srcEnable="video" (a real FCPXML 1.8 DTD attribute) is kept as
        # defense-in-depth, but confirmed 2026-09-13 that it alone does NOT
        # stop Resolve creating a full extra audio track for a clip whose
        # REFERENCED ASSET declares audio - only a dedicated asset_id that
        # itself never declares audio (see picture_asset_ids above) actually
        # prevents it. An <adjust-volume -96dB> gain was tried first and had
        # the same problem for the same reason.
        src_enable_attr = ' srcEnable="video"' if video_only else ""
        ref_id = asset_id if asset_id is not None else asset_ids[media.path]
        attrs = (
            f'ref="{ref_id}" {lane_attr}'
            f'offset="{_time_str(offset_f, fps)}" name="{name}" '
            f'start="{_time_str(start_f, fps)}" '
            f'duration="{_time_str(dur_f, fps)}" '
            f'format="r0"{role_attr}{enabled_attr}{src_enable_attr}'
        )
        inner = ""
        if muted:
            inner += NL + f'{indent}  <adjust-volume amount="-96dB"/>'
        inner += children
        if inner:
            return (f'{indent}<asset-clip {attrs}>{inner}' + NL +
                    f'{indent}</asset-clip>')
        return f'{indent}<asset-clip {attrs}/>'

    # Scene boundaries in frames, and which camera is active at a given
    # frame range's midpoint - only used when camera_media is given. The
    # picture lanes sit above every existing per-speaker lane.
    scene_frames = [(cam, int(round(s * float(fps))), int(round(e * float(fps))))
                    for cam, s, e in (scenes or [])]
    picture_lane_base = len(speaker_media)

    def camera_for_piece(piece_start, piece_end):
        mid = (piece_start + piece_end) / 2
        for cam, s_f, e_f in scene_frames:
            if s_f <= mid < e_f:
                return cam
        return None

    for start_s, end_s in keep_ranges:
        src_start_frames = int(round(start_s * float(fps)))
        src_end_frames = int(round(end_s * float(fps)))
        if src_end_frames - src_start_frames <= 0:
            continue

        extra_cuts = None
        if scene_frames:
            extra_cuts = [f for cam, s_f, e_f in scene_frames
                         for f in (s_f, e_f)]

        pieces = _aligned_pieces(src_start_frames, src_end_frames,
                                 mute_frames_by_speaker, len(speaker_media),
                                 extra_cut_frames=extra_cuts)

        for piece_start, piece_end, muted in pieces:
            piece_dur = piece_end - piece_start
            piece_offset = timeline_cursor_frames + (piece_start - src_start_frames)

            # Connected clips are positioned in the parent clip's own local
            # time, whose origin is the parent's start attribute - so they take
            # the source start, not the timeline offset.
            connected = ""
            for lane, media in enumerate(speaker_media[1:], start=1):
                # offset is in the PARENT's local time, whose origin is the
                # parent's start - so it shifts by the parent's timecode, while
                # start is this speaker's own source in-point and shifts by
                # theirs. The two are only the same number when both recordings
                # carry the same start timecode.
                connected += NL + clip_xml(
                    "                ", media, lane, piece_start + base_tc,
                    piece_start + tc_frames[media.path],
                    piece_dur, muted[lane], roles[lane])

            if camera_media:
                active_camera = camera_for_piece(piece_start, piece_end)
                for cam_index, media in enumerate(camera_media):
                    # Always present, always referencing that camera's
                    # dedicated no-audio asset (picture_asset_ids, never the
                    # audio-carrying asset_ids entry even for V1/V2) - only
                    # `enabled` varies.
                    connected += NL + clip_xml(
                        "                ", media, picture_lane_base + cam_index,
                        piece_start + base_tc, piece_start + tc_frames[media.path],
                        piece_dur, False, None,
                        enabled=(cam_index == active_camera), video_only=True,
                        asset_id=picture_asset_ids[media.path])

            spine_entries.append(clip_xml(
                "            ", speaker_media[0], 0, piece_offset,
                piece_start + base_tc,
                piece_dur, muted[0], roles[0], children=connected))

        timeline_cursor_frames += src_end_frames - src_start_frames

    sequence_duration = _time_str(timeline_cursor_frames, fps)
    spine_body = NL.join(spine_entries)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="{FCPXML_VERSION}">
  <resources>
{chr(10).join(resources)}
  </resources>
  <library>
    <event name="Wavefield">
      <project name="{escape(project_name)}">
        <sequence format="r0" duration="{sequence_duration}" tcStart="0s" tcFormat="NDF" audioLayout="mono" audioRate="{_audio_rate_label(getattr(base, "audio_rate", 48000))}">
          <spine>
{spine_body}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
"""


def write_fcpxml(path, speaker_media, keep_ranges, project_name="Podcast (Wavefield)",
                 mutes=None, scenes=None, camera_media=None):
    xml = build_fcpxml(speaker_media, keep_ranges, project_name, mutes=mutes,
                       scenes=scenes, camera_media=camera_media)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path



