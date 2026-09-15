"""
Guards the shape of TrackChain itself, not its audio behaviour.

describe() spent a while orphaned: it was moved out of the class and left as
dead code inside _isolated_render_worker, so every caller (audio_export, the FX
dialog's preset and apply logging, the chain summary in app.py) raised
AttributeError. The whole suite passed throughout, because nothing asserted the
method was reachable. This does.
"""

import threading

import numpy as np

import vst_host


class _FakeSlot:
    is_native = True

    def __init__(self, name, bypassed=False):
        self.name = name
        self.bypassed = bypassed


def _chain(*slots, enabled=True):
    chain = vst_host.TrackChain()
    chain.enabled = enabled
    chain.slots = list(slots)
    return chain


def test_describe_is_a_trackchain_method():
    assert callable(getattr(vst_host.TrackChain, "describe", None))


def test_describe_empty_chain():
    assert _chain().describe() == "no plugins"


def test_describe_lists_slots_in_order():
    assert _chain(_FakeSlot("Gain"),
                  _FakeSlot("Soap Voice Cleaner")).describe() == (
        "Gain -> Soap Voice Cleaner")


def test_describe_brackets_bypassed_slots():
    assert _chain(_FakeSlot("Gain", bypassed=True)).describe() == "[Gain]"


def test_describe_notes_a_disabled_chain():
    assert _chain(_FakeSlot("Gain"), enabled=False).describe() == (
        "Gain  (chain off)")


class _ProcessingSlot(_FakeSlot):
    """A native slot that actually runs, for process_slots() tests."""

    def __init__(self, name, on_call):
        super().__init__(name)
        self.lock = threading.Lock()
        self._on_call = on_call

    def process(self, audio, sample_rate, reset=False):
        self._on_call(self.name)
        return audio


def test_process_slots_stops_between_plugins_when_cancelled():
    """
    should_cancel is checked once per plugin (between whole-track passes),
    never mid-plugin-call - see process_slots()'s docstring. A cancel
    tripped after the first plugin must stop the remaining plugins in the
    chain from running at all, while still returning the audio processed so
    far rather than discarding it.
    """
    calls = []
    slots = [_ProcessingSlot(f"plugin{i}", calls.append) for i in range(3)]
    chain = _chain(*slots)
    audio = np.zeros(10, dtype=np.float32)

    result = chain.process_slots(
        audio, 48000, chain.slots, should_cancel=lambda: len(calls) >= 1)

    assert calls == ["plugin0"]
    assert result.shape == audio.shape


def test_process_slots_runs_every_plugin_when_never_cancelled():
    calls = []
    slots = [_ProcessingSlot(f"plugin{i}", calls.append) for i in range(3)]
    chain = _chain(*slots)
    audio = np.zeros(10, dtype=np.float32)

    chain.process_slots(audio, 48000, chain.slots, should_cancel=lambda: False)

    assert calls == ["plugin0", "plugin1", "plugin2"]


