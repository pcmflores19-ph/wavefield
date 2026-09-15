"""
`AutoCutApp._track_render_progress` (2026-09-14) replaces two duplicated,
buggy progress closures - `_render_mix` (video export's audio-render phase)
and `_export_audio_worker` (the WAV export) - that each computed a fraction
of `(index + 0.5) / track_count`. That assumed every speaker track costs the
same to decode+process, which is wrong the moment tracks have very different
lengths or VST chains, and got actively worse once `render_tracks` started
running multiple tracks concurrently: the fraction could visibly jump
backward if a later track's message reached the callback before an earlier
one's, since the old logic only knew "which track is this message about",
not how much of the whole job was genuinely done.

These tests exercise the replacement directly against a minimal stub, the
same pattern tests/test_peaks_redraw_cache.py already uses for AutoCutApp
methods that don't need a real Tk root.
"""

import app as app_module


class _StubApp:
    """Just enough of AutoCutApp for _track_render_progress to run unmodified."""

    def __init__(self, speaker_paths, durations):
        self.speaker_paths = speaker_paths
        self._audio_durations = durations
        self.log = lambda *a, **k: None
        self.steps = []

    def _export_step(self, text=None, fraction=None):
        self.steps.append((text, fraction))


def _fractions(stub):
    return [fraction for _text, fraction in stub.steps]


def test_fraction_is_weighted_by_real_duration_not_track_count():
    # Track 0 is 10s, track 1 is 90s - track 0 finishing should count for
    # far less than half, unlike the old (index + 0.5) / total formula.
    stub = _StubApp(["short.wav", "long.wav"], [10.0, 90.0])
    progress = app_module.AutoCutApp._track_render_progress(stub)

    progress("Rendering short.wav...")
    progress("finished short.wav")

    last_fraction = stub.steps[-1][1]
    assert last_fraction < 0.15, (
        f"finishing the 10s track out of 100s total should read close to "
        f"0.1, not {last_fraction} (a count-based fraction would show 0.5)")


def test_fraction_reaches_one_once_every_track_finishes():
    stub = _StubApp(["a.wav", "b.wav"], [30.0, 70.0])
    progress = app_module.AutoCutApp._track_render_progress(stub)

    progress("Rendering a.wav...")
    progress("decoding a.wav")
    progress("finished a.wav")
    progress("Rendering b.wav...")
    progress("processing b.wav through Compressor")
    progress("finished b.wav")

    assert stub.steps[-1][1] == 1.0


def test_fraction_never_decreases_even_out_of_order():
    """
    render_tracks runs tracks concurrently - nothing guarantees track 0's
    messages all arrive before track 1's. The old index-based fraction
    could jump backward here; the duration-weighted stage sum must not.
    """
    stub = _StubApp(["a.wav", "b.wav", "c.wav"], [20.0, 20.0, 20.0])
    progress = app_module.AutoCutApp._track_render_progress(stub)

    # b (index 1) reports first and finishes before a or c are even seen.
    progress("Rendering b.wav...")
    progress("finished b.wav")
    progress("Rendering a.wav...")
    progress("decoding a.wav")
    progress("Rendering c.wav...")
    progress("finished a.wav")
    progress("finished c.wav")

    fractions = _fractions(stub)
    for earlier, later in zip(fractions, fractions[1:]):
        assert later >= earlier, (
            f"fraction went backward: {earlier} -> {later} in {fractions}")
    assert fractions[-1] == 1.0


def test_unmatched_message_keeps_last_fraction_instead_of_none():
    """
    A message that doesn't name any track (e.g. "mix peaked...") used to
    force fraction=None, flipping the dialog to an indeterminate spinner
    mid-export. It should now just keep reporting the last real fraction.
    """
    stub = _StubApp(["a.wav"], [10.0])
    progress = app_module.AutoCutApp._track_render_progress(stub)

    progress("Rendering a.wav...")
    progress("finished a.wav")
    progress("mix peaked at 1.10, limiting overs")

    assert stub.steps[-1][1] == 1.0
    assert all(fraction is not None for fraction in _fractions(stub))


def test_finished_pins_stage_even_if_reported_message_order_is_odd():
    """A track already marked finished must not be knocked back down by a
    stray later message that happens to still name it."""
    stub = _StubApp(["a.wav"], [10.0])
    progress = app_module.AutoCutApp._track_render_progress(stub)

    progress("finished a.wav")
    progress("decoding a.wav")  # e.g. a delayed/duplicated log line

    assert stub.steps[-1][1] == 1.0


def test_scale_compresses_the_reported_fraction():
    """_export_audio_worker reserves the last 10% of the bar for writing
    files by passing scale=0.9."""
    stub = _StubApp(["a.wav"], [10.0])
    progress = app_module.AutoCutApp._track_render_progress(stub, scale=0.9)

    progress("finished a.wav")

    assert stub.steps[-1][1] == 0.9


def test_falls_back_to_equal_weights_if_durations_are_stale():
    """self._audio_durations can be out of sync with self.speaker_paths
    (e.g. tracks changed since the last analysis) - must not crash or
    misindex, just fall back to equal weighting."""
    stub = _StubApp(["a.wav", "b.wav"], [42.0])  # length mismatch
    progress = app_module.AutoCutApp._track_render_progress(stub)

    progress("finished a.wav")
    progress("finished b.wav")

    assert stub.steps[-1][1] == 1.0


def test_message_carries_no_eta_text():
    """
    An earlier version of this callback appended an "about Xs left"
    estimate to the message, computed fresh from elapsed/fraction on every
    call. A healthy VST chain can go a long time with no progress message
    at all (vst_host.py's process_slots only calls `log` on an error/
    chunk-fallback/mismatch, never on the ordinary per-plugin success
    path, confirmed 2026-09-14), so that estimate could read back wildly
    inflated - e.g. "5s left" after 10s had already passed. Rather than
    keep guessing, this callback reports only a percentage, never a time
    estimate.
    """
    stub = _StubApp(["a.wav"], [10.0])
    progress = app_module.AutoCutApp._track_render_progress(stub)

    progress("decoding a.wav")

    text = stub.steps[-1][0]
    assert "left" not in text
    assert text == "decoding a.wav"
