"""
FCPXML timeline export.

The golden-file test pins build_fcpxml's current output for a small,
representative fixture (2 speakers, a couple of keep ranges, a mute) - the
scene-switching feature adds new optional parameters (scenes, camera_media)
that must be a no-op on existing callers. If this test ever needs updating
for a deliberate change to the NON-scene-switching output, regenerate it
deliberately - don't just paste in whatever the new output happens to be.
"""

from fractions import Fraction

import fcpxml_writer
from media_probe import MediaInfo


def _media(path, duration, fps=Fraction(30, 1), width=1920, height=1080):
    return MediaInfo(path, fps, width, height, duration, True,
                     audio_channels=1, audio_rate=48000)


def _fixture():
    speaker_media = [
        _media("/recordings/host.mp4", 20.0),
        _media("/recordings/guest.mp4", 20.0),
    ]
    keep_ranges = [(0.0, 8.0), (10.0, 18.0)]
    mutes = [(1, 3.0, 4.0)]
    return speaker_media, keep_ranges, mutes


def test_build_fcpxml_unchanged_without_scene_data():
    """
    Pinned baseline: no scenes/camera_media given (every pre-existing caller)
    must keep producing exactly this output. Two keep ranges split into 4
    pieces by the one mute boundary pair - a scene-switching regression
    would silently inflate this piece/lane count.
    """
    speaker_media, keep_ranges, mutes = _fixture()
    xml = fcpxml_writer.build_fcpxml(speaker_media, keep_ranges,
                                     project_name="Golden Fixture", mutes=mutes)

    assert '<asset id="r1"' in xml
    assert '<asset id="r2"' in xml
    assert "enabled=" not in xml
    assert xml.count("<asset-clip") == 8
    assert xml.count('lane="1"') == 4
    assert xml.count("adjust-volume") == 1


def test_build_fcpxml_scene_data_is_opt_in():
    """
    Passing scenes/camera_media as None (the default) or omitting them
    entirely must produce identical output - the new parameters must not
    change behavior unless BOTH are actually supplied.
    """
    speaker_media, keep_ranges, mutes = _fixture()

    default_xml = fcpxml_writer.build_fcpxml(
        speaker_media, keep_ranges, project_name="Golden Fixture", mutes=mutes)
    explicit_none_xml = fcpxml_writer.build_fcpxml(
        speaker_media, keep_ranges, project_name="Golden Fixture", mutes=mutes,
        scenes=None, camera_media=None)

    assert default_xml == explicit_none_xml


def test_build_fcpxml_requires_scenes_and_camera_media_together():
    speaker_media = [_media("/recordings/host.mp4", 20.0),
                     _media("/recordings/guest.mp4", 20.0)]
    try:
        fcpxml_writer.build_fcpxml(speaker_media, [(0.0, 10.0)],
                                   scenes=[(0, 0.0, 10.0)])
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_build_fcpxml_scene_switching_adds_static_reference_picture_lanes():
    """
    Each camera's picture lane always references the SAME asset for every
    piece - only `enabled` toggles - which is the whole point of this
    design (see build_fcpxml's docstring). V1/V2's picture lanes reference a
    SEPARATE, video-only asset from the audio-carrying r1/r2 used by the
    spine/speaker lanes - confirmed by direct Resolve import that reusing
    the audio-carrying asset (even with srcEnable="video"/adjust-volume)
    still left Resolve creating extra duplicate audio tracks.
    """
    host = _media("/recordings/host.mp4", 20.0)
    guest = _media("/recordings/guest.mp4", 20.0)
    v3 = _media("/recordings/merged.mp4", 20.0)
    speaker_media = [host, guest]
    camera_media = [host, guest, v3]
    # One scene switch at 5s, inside a single 0-10s keep range.
    scenes = [(0, 0.0, 5.0), (1, 5.0, 10.0)]

    xml = fcpxml_writer.build_fcpxml(speaker_media, [(0.0, 10.0)],
                                     scenes=scenes, camera_media=camera_media)

    # 5 assets: host/guest's audio-carrying originals (r1/r2) PLUS a
    # dedicated video-only asset per camera for the picture lanes (r3/r4/r5)
    # - every picture-lane asset declares no audio at all.
    assert xml.count("<asset id=") == 5
    assert xml.count("hasAudio=") == 2

    # The scene boundary splits the single keep range into 2 pieces, each
    # gaining 3 picture-lane clips (lanes 2/3/4, above the existing
    # per-speaker lane 1) - 6 picture clips total, each video-only.
    assert xml.count('lane="2"') == 2
    assert xml.count('lane="3"') == 2
    assert xml.count('lane="4"') == 2
    # One camera enabled and two disabled per piece -> 4 disabled total.
    assert xml.count('enabled="0"') == 4
    # Every picture-lane clip excludes audio via srcEnable="video" (not an
    # adjust-volume gain, which still leaves a full duplicate silent audio
    # track behind - confirmed by direct Resolve import, see fcpxml_writer.py).
    assert xml.count('srcEnable="video"') == 6
    assert "adjust-volume" not in xml
