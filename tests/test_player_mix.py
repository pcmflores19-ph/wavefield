"""
How Player._mix_into runs a track's effect chain.

Live-only architecture: there is no pre-rendered/baked alternative any more
(see docs/DEVELOPERS.md) - the whole chain runs on every block, like OBS
Studio's filter chain on a stream. These cover that it always runs live, and
that the `reset` flag (True only on the block right after a position
discontinuity - a seek, or the callback loop skipping a cut) actually
reaches the chain and changes behavior for a stateful effect, instead of
being silently ignored.
"""

import numpy as np
import pytest

import effects
import player
import vst_host


@pytest.fixture
def track():
    """A 3-second track with an empty chain, ready for a caller to add
    slots to."""
    count = player.SAMPLE_RATE * 3
    source = np.random.default_rng(5).integers(
        -20000, 20000, count, dtype=np.int16)
    handle = player.Track("spk", source, "source.wav")
    handle.chain = vst_host.TrackChain()
    return handle


def mix_block(handle, frames=1024, start_sample=0, reset=False):
    out = np.zeros(frames, dtype=np.float32)
    player.Player()._mix_into(out, start_sample, frames, [handle],
                              reset=reset)
    return out


def test_chain_runs_live_every_block(track):
    track.chain.add_native("gain")
    assert mix_block(track).any()


def test_disabled_chain_is_not_run(track):
    track.chain.add_native("gain")
    track.chain.enabled = False

    with_chain_off = mix_block(track)
    raw = np.asarray(track.samples[:1024]).astype(np.float32) / 32768.0

    assert np.array_equal(with_chain_off, raw)


def test_empty_chain_does_not_error(track):
    # No slots at all - _mix_into must skip straight to the raw samples,
    # not try to process through an empty list.
    raw = np.asarray(track.samples[:1024]).astype(np.float32) / 32768.0
    assert np.array_equal(mix_block(track), raw)


def test_mix_into_sets_track_peak_level_from_true_peak(track):
    # A sub-Nyquist sine, phased so no discrete sample lands exactly on its
    # peak (same genuine inter-sample-overshoot construction as
    # test_effects.py's true_peak tests) - so the true-peak reading is
    # provably higher than a plain sample-peak read of the same mixed
    # output. peak_level feeds both the main mixer's per-track meter and the
    # effects/plugin-chain dialog meter (fx_dialog.py), so this one value
    # covers both.
    n = player.SAMPLE_RATE * 3
    f = 0.3 * player.SAMPLE_RATE / 2
    t = np.arange(n) / player.SAMPLE_RATE
    track.samples = (32000 * np.sin(2 * np.pi * f * t + 0.37)).astype(np.int16)

    out = mix_block(track)
    sample_peak = float(np.abs(out).max())

    assert track.peak_level > sample_peak
    assert track.peak_level == pytest.approx(effects.true_peak(out))


def test_callback_master_peak_uses_true_peak(track, monkeypatch):
    calls = []
    real_true_peak = effects.true_peak

    def spy(samples, *args, **kwargs):
        calls.append(samples)
        return real_true_peak(samples, *args, **kwargs)

    monkeypatch.setattr(effects, "true_peak", spy)

    p = player.Player()
    p.tracks = [track]
    p.duration = track.samples.size / player.SAMPLE_RATE
    p.edited_mode = False

    outdata = np.zeros((1024, 1), dtype=np.float32)
    p._callback(outdata, 1024, None, None)

    assert calls, "Player._callback did not call effects.true_peak"
    assert p.master_peak == pytest.approx(real_true_peak(calls[-1]))


def test_position_is_compensated_for_output_latency():
    """
    Displayed position must be ahead-of-written-sample minus the output
    stream's own latency, not the raw written-sample index - before this
    fix, `position` never accounted for the gap between "handed to the
    callback" and "actually reaches the speakers", so the drawn playhead
    always read ahead of what was actually audible.
    """
    p = player.Player()
    p._pos = int(2.0 * player.SAMPLE_RATE)   # 2.0s of audio already written
    p._output_latency = 0.05                 # 50ms of output-stream latency

    assert p.position == pytest.approx(1.95)


def test_seek_is_never_latency_adjusted():
    """seek() must land exactly where asked - applying the same latency
    correction there (rather than only to what's displayed) would make a
    seek-then-read round trip drift, and repeated skips would compound it."""
    p = player.Player()
    p.duration = 10.0
    p._output_latency = 0.05

    p.seek(3.0)

    assert p._pos == int(3.0 * player.SAMPLE_RATE)


def test_callback_tolerates_missing_time_info(track):
    """time_info can be None (as in this suite's other _callback calls) or
    lack the DAC-timing fields sounddevice sometimes doesn't populate -
    either must fall back to zero latency, not raise (a raise other than the
    normal end-of-track CallbackStop, which is unrelated pre-existing
    behavior for an empty/exhausted track)."""
    import sounddevice as sd

    p = player.Player()
    p.tracks = [track]
    p.duration = track.samples.size / player.SAMPLE_RATE
    p.edited_mode = False

    outdata = np.zeros((1024, 1), dtype=np.float32)
    p._callback(outdata, 1024, None, None)

    assert p._output_latency == 0.0

    # Confirm the pre-existing CallbackStop path (nothing left to play) is
    # unaffected by the latency capture running before it every block.
    p.tracks = []
    p.duration = 0.0
    with pytest.raises(sd.CallbackStop):
        p._callback(outdata, 1024, None, None)
    assert p._output_latency == 0.0


def test_reset_reaches_the_chain_and_changes_the_result(track):
    """The actual bug this architecture depends on being fixed: without
    `reset` reaching the chain, a real discontinuity (a seek) would have no
    effect on a stateful effect's sound. Play one block normally, then a
    SECOND block at the same source position either as a continuation
    (reset=False, as a normal next callback block would be) or as a fresh
    start (reset=True, as the block right after a seek is) - these must
    differ, proving the flag really reaches vst_host.NativeSlot's live
    state instead of being accepted and dropped."""
    track.chain.add_native("compressor")
    mix_block(track, reset=True)   # first block, establishes envelope state

    continuation = mix_block(track, reset=False)
    fresh_start = mix_block(track, reset=True)

    assert not np.array_equal(continuation, fresh_start)
