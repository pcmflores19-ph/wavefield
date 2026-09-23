"""
Plugins that refuse mono (PodcastPlugins TRACK/MASTER, DReverb, ...) must still
process Wavefield's mono tracks - see channel_adapt.py.
"""
import types

import numpy as np
import pytest

import channel_adapt
import vst_host


class _FixedBus:
    """Stands in for a pedalboard plugin whose main bus has a fixed width."""

    def __init__(self, channels, gain=0.5):
        self.channels = channels
        self.gain = gain
        self.calls = 0

    def __call__(self, audio, sample_rate, reset=False):
        self.calls += 1
        if audio.shape[0] != self.channels:
            raise ValueError(
                f"Plugin 'Stub' does not support {audio.shape[0]}-channel "
                f"output. (Main bus currently expects {self.channels} input "
                f"channels and {self.channels} output channels.)")
        return audio * self.gain


def _mono(n=1000):
    return np.linspace(-0.5, 0.5, n, dtype=np.float32).reshape(1, -1)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_process_mono_matches_any_fixed_bus(width):
    plugin, state = _FixedBus(width), types.SimpleNamespace(channels=1)
    audio = _mono()
    out = channel_adapt.process_mono(plugin, state, audio, 44100, True)
    assert out.shape == audio.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, audio * 0.5, rtol=1e-6)
    assert state.channels == width


def test_width_is_learned_once_not_retried_every_block():
    plugin, state = _FixedBus(2), types.SimpleNamespace(channels=1)
    for _ in range(5):
        channel_adapt.process_mono(plugin, state, _mono(), 44100, False)
    assert plugin.calls == 1 + 5      # one rejected mono try, then 5 stereo


def test_unrelated_value_error_still_propagates():
    def broken(audio, sample_rate, reset=False):
        raise ValueError("block size 99999999 is too large")
    with pytest.raises(ValueError):
        channel_adapt.process_mono(broken, types.SimpleNamespace(channels=1),
                                   _mono(), 44100, False)


def test_required_input_channels_parses_the_pedalboard_message():
    exc = ValueError("Plugin 'X' does not support 1-channel output. (Main bus "
                     "currently expects 2 input channels and 2 output channels.)")
    assert channel_adapt.required_input_channels(exc) == 2
    assert channel_adapt.required_input_channels(ValueError("nope")) is None
    assert channel_adapt.required_input_channels(RuntimeError("channel")) is None


def _chain_with(plugin):
    chain = vst_host.TrackChain()
    slot = vst_host.PluginSlot("Stub", "stub.vst3", plugin)
    chain.slots.append(slot)
    return chain, slot


def test_chain_applies_stereo_only_plugin_to_mono_track():
    chain, slot = _chain_with(_FixedBus(2))
    audio = _mono().reshape(-1)
    out = chain.process(audio, 44100, reset=True)
    assert out.shape == audio.shape
    np.testing.assert_allclose(out, audio * 0.5, rtol=1e-6)
    assert slot.channels == 2 and slot.last_error is None


def test_chunked_retry_also_adapts():
    """The 30 s chunk fallback calls the plugin separately from the first try."""
    calls = {"n": 0}
    inner = _FixedBus(2)

    def flaky(audio, sample_rate, reset=False):
        calls["n"] += 1
        if audio.shape[0] == 2 and audio.shape[1] > 500:
            raise ValueError("block too large")     # forces the chunked path
        return inner(audio, sample_rate, reset)

    chain, slot = _chain_with(flaky)
    audio = _mono(1200).reshape(-1)
    out = chain.process(audio, 10, reset=True)      # 10 Hz -> 300-sample chunks
    # Whole-buffer stereo attempt fails, then every chunk is retried; the
    # important part is that the mono track still comes back processed.
    np.testing.assert_allclose(out, audio * 0.5, rtol=1e-6)


def test_skipped_plugin_is_recorded_not_silent():
    def broken(audio, sample_rate, reset=False):
        raise RuntimeError("plugin exploded")
    chain, slot = _chain_with(broken)
    audio = _mono().reshape(-1)
    out = chain.process(audio, 44100, reset=True)
    np.testing.assert_array_equal(out, audio)       # passed through
    assert "plugin exploded" in slot.last_error
