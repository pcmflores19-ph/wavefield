"""
_compute_mutes must clamp each lane's mute ranges to that lane's own real
decoded-audio duration, never a single shared cross-track value.

Before this fix, every lane's "everywhere nobody's talking" mute computation
was clamped to one shared duration (the longest track's ffprobe-reported
length) passed in as a single scalar. A shorter lane's waveform correctly
stopped drawing where its real audio ended, but its mute overlay kept
extending past that point into a region with no real audio at all - because
compute_auto_mutes_from_intervals had no speech to complement there, and the
shared clamp said the timeline kept going.
"""

import app as app_module


class _StubApp:
    """Just enough of AutoCutApp for _compute_mutes to run unmodified."""

    def __init__(self):
        self.log = lambda *a, **k: None


def test_shorter_lanes_mute_range_is_clamped_to_its_own_duration():
    stub = _StubApp()

    # Two lanes, both silent throughout (no detected speech) - the simplest
    # case that produces a single "everywhere is inactive" mute range per
    # lane, bounded only by whatever duration is passed in for that lane.
    speech_per_speaker = [[], []]
    long_duration = 100.0
    short_duration = 40.0

    mutes = app_module.AutoCutApp._compute_mutes(
        stub, speech_per_speaker, None, None, [long_duration, short_duration])

    assert mutes[0] == [(0.0, long_duration)]
    assert mutes[1] == [(0.0, short_duration)]
    # The actual bug: the short lane's mute must never reach past its own
    # real duration just because another lane runs longer.
    assert mutes[1][0][1] <= short_duration


def test_falls_back_to_first_duration_when_list_is_short():
    """Defensive fallback, matching _draw_waveform's own pattern - a lane
    index past the end of `durations` should not crash."""
    stub = _StubApp()

    mutes = app_module.AutoCutApp._compute_mutes(
        stub, [[], []], None, None, [55.0])

    assert mutes[0] == [(0.0, 55.0)]
    assert mutes[1] == [(0.0, 55.0)]
