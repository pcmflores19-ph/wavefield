"""
_apply_analysis_results commits one analysis pass as a single step.

Before this fix, _analyze_worker (a background thread) wrote 11 separate
instance attributes directly, one statement at a time, with no lock and no
main-thread marshalling until AFTER all of them - the periodic redraw
(_tick, every 60ms on the main thread) could land between two of those
statements and read a mix of one pass's peaks paired with a different pass's
duration. The fix moves the whole commit into one method, invoked via
root.after from the background thread, so the main thread only ever sees the
fields before or after a complete update, never mid-update.
"""

import app as app_module


class _StubApp:
    """Just enough of AutoCutApp for _apply_analysis_results to run
    unmodified - _analysis_done itself is heavy (rebuilds UI widgets), so
    it's stubbed out and just recorded as called."""

    def __init__(self):
        self.analysis_done_called = False
        self.apply_edits_saw = None

    def _apply_edits(self):
        # Pushes the auto-mutes just committed into the player's tracks; must
        # therefore run after auto_mutes is set (see the test below).
        self.apply_edits_saw = self.auto_mutes

    def _analysis_done(self):
        self.analysis_done_called = True


def _payload(**overrides):
    base = {
        "speaker_media": ["media"],
        "audio_durations": [12.5, 8.0],
        "per_speaker_speech": [[(0.0, 1.0)], [(0.5, 2.0)]],
        "speech_levels": [[1.0], [2.0]],
        "speech_hop": 0.02,
        "timeline_duration": 12.5,
        "peaks_list": [[0.1, 0.2], [0.3, 0.4]],
        "auto_mutes": [[(1.0, 12.5)], [(2.0, 8.0)]],
        "view_span": 12.5,
    }
    base.update(overrides)
    return base


def test_all_fields_land_together():
    stub = _StubApp()
    results = _payload()

    app_module.AutoCutApp._apply_analysis_results(stub, results)

    assert stub.speaker_media == results["speaker_media"]
    assert stub._audio_durations == results["audio_durations"]
    assert stub.per_speaker_speech == results["per_speaker_speech"]
    assert stub._speech_levels == results["speech_levels"]
    assert stub._speech_hop == results["speech_hop"]
    assert stub.timeline_duration == results["timeline_duration"]
    assert stub.peaks_list == results["peaks_list"]
    assert stub.auto_mutes == results["auto_mutes"]
    assert stub.playhead is None
    assert stub.view_start == 0.0
    assert stub.view_span == results["view_span"]


def test_analysis_done_runs_after_the_commit():
    """_analysis_done (and anything it triggers, like a redraw) must only
    ever run once every field above is already set - by the time it's
    reached, peaks_list/_audio_durations/timeline_duration must already be
    self-consistent, which this checks from inside the stubbed callback."""
    stub = _StubApp()
    seen = {}

    def _analysis_done(self=stub):
        # Snapshot what a redraw triggered from here would actually see.
        seen["peaks_list"] = stub.peaks_list
        seen["audio_durations"] = stub._audio_durations
        seen["timeline_duration"] = stub.timeline_duration

    stub._analysis_done = _analysis_done
    results = _payload()

    app_module.AutoCutApp._apply_analysis_results(stub, results)

    assert seen["peaks_list"] == results["peaks_list"]
    assert seen["audio_durations"] == results["audio_durations"]
    assert seen["timeline_duration"] == results["timeline_duration"]


def test_new_auto_mutes_are_pushed_to_the_player_after_they_are_committed():
    """A fresh analysis pass must reach playback: _apply_edits() is the only
    thing that copies mute ranges into the player's tracks, so it has to run
    (after auto_mutes is set, before _analysis_done redraws)."""
    stub = _StubApp()
    results = _payload()

    app_module.AutoCutApp._apply_analysis_results(stub, results)

    assert stub.apply_edits_saw == results["auto_mutes"]
