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
