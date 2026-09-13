"""
Guards the shape of TrackChain itself, not its audio behaviour.

describe() spent a while orphaned: it was moved out of the class and left as
dead code inside _isolated_render_worker, so every caller (audio_export, the FX
dialog's preset and apply logging, the chain summary in app.py) raised
AttributeError. The whole suite passed throughout, because nothing asserted the
method was reachable. This does.
"""

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


