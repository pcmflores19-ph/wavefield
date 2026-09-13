"""
Live playback runs the effect chain on every ~21ms audio callback block
(player.py), not in one pass like export. That's only safe if each native
effect's internal state (envelope/gate/filter memory) carries across
successive blocks instead of resetting on every single call - otherwise a
continuously playing compressor/gate/EQ would sound different (and wrong)
during live monitoring than what export produces. These pin that property
down directly, rather than trusting the docstrings alone. See
vst_host.NativeSlot's `_live_state` and TrackChain.process_slots' `reset`
threading.
"""

import threading

import numpy as np
import pytest

import effects
import vst_host


def _test_signal(sample_rate=48000, seconds=3):
    rng = np.random.RandomState(0)
    n = sample_rate * seconds
    samples = (rng.randn(n).astype(np.float32) * 0.3)
    # Loud transients partway through, so a chunk boundary can land
    # mid-attack or mid-release rather than only during quiet audio.
    samples[sample_rate:sample_rate + 1000] *= 5.0
    samples[2 * sample_rate:2 * sample_rate + 500] *= 8.0
    return samples


@pytest.mark.parametrize("key,params", [
    ("compressor", {}),
    ("limiter", {}),
    ("expander", {}),
    ("noise_gate", {}),
    ("eq3", {"low_db": 6.0, "mid_db": -4.0, "high_db": 3.0}),
    ("gain", {"gain_db": -3.0}),
])
def test_chunked_state_matches_single_call(key, params):
    sample_rate = 48000
    samples = _test_signal(sample_rate)

    whole = effects.apply(key, samples, sample_rate, params)

    # An awkward split, not aligned to any internal chunk size, landing
    # inside the first loud transient.
    split = sample_rate + 777
    state = {}
    part1 = effects.apply(key, samples[:split], sample_rate, params,
                          state=state)
    part2 = effects.apply(key, samples[split:], sample_rate, params,
                          state=state)
    chunked = np.concatenate([part1, part2])

    assert np.array_equal(whole, chunked)


def test_chunked_state_matches_single_call_across_many_small_blocks():
    """Not just one split - many ~1024-sample blocks, the real audio
    callback's blocksize (player.py), including one landing inside the
    second transient."""
    sample_rate = 48000
    samples = _test_signal(sample_rate)
    params = {}

    whole = effects.apply("compressor", samples, sample_rate, params)

    block = 1024
    state = {}
    pieces = []
    for start in range(0, samples.size, block):
        stop = min(start + block, samples.size)
        pieces.append(effects.apply("compressor", samples[start:stop],
                                    sample_rate, params, state=state))
    chunked = np.concatenate(pieces)

    assert np.array_equal(whole, chunked)


def test_resetting_every_block_actually_differs():
    """Proves the above tests are meaningful: resetting state on every block
    (the bug this fixes) produces real, substantial drift from one
    continuous pass - not a vacuous pass because floating point noise is
    already near zero."""
    sample_rate = 48000
    samples = _test_signal(sample_rate)
    params = {}

    whole = effects.apply("compressor", samples, sample_rate, params)

    block = 1024
    pieces = []
    for start in range(0, samples.size, block):
        stop = min(start + block, samples.size)
        # No state carried - fresh internal state every call, the old bug.
        pieces.append(effects.apply("compressor", samples[start:stop],
                                    sample_rate, params))
    reset_every_block = np.concatenate(pieces)

    assert not np.allclose(whole, reset_every_block, atol=1e-4)


# The real NativeSlot (not a fake): needs its actual .process() method and
# ._live_state, which is what process_slots' native branch and Player's live
# callback both depend on - a fake missing those would fail silently inside
# process_slots' per-slot try/except and make the "baseline" secretly a
# no-op.
_FakeNativeSlot = vst_host.NativeSlot


class _FakePluginSlot:
    """Stands in for a VST3 PluginSlot - pedalboard plugin objects carry
    their own internal state across successive calls automatically (not
    something this codebase manages), so the only thing to verify here is
    that `reset` is threaded through correctly, never that it's chunked."""
    is_native = False

    def __init__(self, name="FakeVST"):
        self.name = name
        self.bypassed = False
        self.lock = threading.Lock()
        self.calls = []

    def plugin(self, buf, sample_rate, reset=False):
        self.calls.append((buf.shape[1], reset))
        return buf * 0.5   # some audible, deterministic transform


def test_process_slots_live_blocks_match_one_continuous_pass():
    """The actual live-playback shape: many small process_slots() calls,
    reset=True only on the very first one - must equal one process() call
    over the whole buffer, for a chain of only native effects."""
    sample_rate = 48000
    samples = _test_signal(sample_rate)

    chain = vst_host.TrackChain()
    chain.slots = [_FakeNativeSlot("compressor"),
                  _FakeNativeSlot("gain", {"gain_db": -6.0})]

    straight = chain.process(samples.copy(), sample_rate, reset=True)

    block = 1024
    pieces = []
    for i, start in enumerate(range(0, samples.size, block)):
        stop = min(start + block, samples.size)
        pieces.append(chain.process_slots(
            samples[start:stop].copy(), sample_rate, chain.slots,
            reset=(i == 0)))
    live = np.concatenate(pieces)

    assert np.array_equal(straight, live)


def test_process_slots_passes_reset_through_to_a_plugin_slot():
    """A VST3 slot is never chunked - it always sees the whole buffer handed
    to process_slots in one call - but it must still see `reset` correctly,
    so pedalboard's own internal state resets on a real discontinuity (a
    seek) the same way a native slot's _live_state does."""
    sample_rate = 48000
    samples = _test_signal(sample_rate, seconds=1)

    plugin_slot = _FakePluginSlot()
    chain = vst_host.TrackChain()
    chain.slots = [_FakeNativeSlot("gain", {"gain_db": -3.0}), plugin_slot]

    chain.process_slots(samples.copy(), sample_rate, chain.slots, reset=True)
    chain.process_slots(samples.copy(), sample_rate, chain.slots, reset=False)

    assert len(plugin_slot.calls) == 2
    assert plugin_slot.calls[0] == (samples.size, True)
    assert plugin_slot.calls[1] == (samples.size, False)


def test_native_slot_state_persists_until_reset():
    """NativeSlot itself, directly: state carries block to block, and
    reset=True clears it - the two halves of the live-playback contract."""
    sample_rate = 48000
    samples = _test_signal(sample_rate)
    slot = vst_host.NativeSlot("compressor")

    block = 1024
    live_pieces = []
    for start in range(0, samples.size, block):
        stop = min(start + block, samples.size)
        live_pieces.append(slot.process(samples[start:stop].copy(),
                                        sample_rate, reset=(start == 0)))
    live = np.concatenate(live_pieces)

    fresh_slot = vst_host.NativeSlot("compressor")
    straight = fresh_slot.process(samples.copy(), sample_rate, reset=True)

    assert np.array_equal(straight, live)

    # A reset mid-stream restarts state, so the block right after a
    # discontinuity does NOT equal the same block played as an
    # uninterrupted continuation of what came before it - a seek must
    # actually change the sound, not be silently ignored.
    continuous_slot = vst_host.NativeSlot("compressor")
    continuous_slot.process(samples[:block].copy(), sample_rate, reset=True)
    continuous_next = continuous_slot.process(
        samples[block:2 * block].copy(), sample_rate, reset=False)

    discontinuous_slot = vst_host.NativeSlot("compressor")
    discontinuous_slot.process(samples[:block].copy(), sample_rate, reset=True)
    discontinuous_next = discontinuous_slot.process(
        samples[block:2 * block].copy(), sample_rate, reset=True)

    assert not np.array_equal(continuous_next, discontinuous_next)
