"""
Running mono audio through plugins that do not take mono.

Wavefield's tracks are mono, but a VST3 declares a fixed main-bus layout and
pedalboard refuses any other channel count with a ValueError such as

    Plugin 'PodcastPlugins TRACK' does not support 1-channel output.
    (Main bus currently expects 2 input channels and 2 output channels.)

DAWs get past this by presenting the plugin with the layout it asks for. This
does the same: try mono first, and if the plugin names a different input
channel count, feed it that many copies of the signal and average whatever
comes back down to one channel. Identical channels mean any L/R-linked
detection sees exactly the mono signal; only stereo-width effects are lost,
which a mono track has no use for.

Used by vst_host.TrackChain (playback and renders) and by plugin_editor (the
meter feed in the editor subprocess). Kept free of other Wavefield imports so
the editor subprocess can use it without pulling in the rest of the app.
"""

import re

import numpy

_EXPECTS = re.compile(r"expects\s+(\d+)\s+input\s+channels?", re.IGNORECASE)


def required_input_channels(exc):
    """
    The input channel count a plugin asks for in a "does not support N-channel"
    error, or None if `exc` is not that kind of error.
    """
    if not isinstance(exc, ValueError) or "channel" not in str(exc):
        return None
    match = _EXPECTS.search(str(exc))
    return int(match.group(1)) if match else None


def _call(plugin, audio, sample_rate, reset):
    """
    `plugin(audio, sample_rate, reset=reset)`, tolerating a plugin that can only
    reset on the main thread.

    Some VST3s (PodcastPlugins MASTER) can only be reset by reloading them, and
    pedalboard allows that on the main thread alone; from any other thread -
    the audio callback, the editor window's feed thread - `reset=True` raises
    "must be reloaded on the main thread ... pass reset=False". Without this
    the editor's feed died on its very first block (the window's meters never
    moved and its switches never reached the plugin), and live playback
    dropped the master for the first block after every seek. The audio still
    has to be processed, so it goes through without the reset: the plugin
    keeps its previous internal state instead of starting fresh, which is far
    better than not processing at all.
    """
    try:
        return plugin(audio, sample_rate, reset=reset)
    except RuntimeError as exc:
        if reset and "main thread" in str(exc):
            return plugin(audio, sample_rate, reset=False)
        raise


def process_mono(plugin, state, audio, sample_rate, reset):
    """
    `plugin(audio, sample_rate, reset=reset)` for mono `audio` of shape (1, n),
    returning shape (1, n) unless the plugin changes the length.

    `state` is any object with an int attribute `channels`, starting at 1. It
    remembers how many copies this plugin needs, so a stereo-only plugin costs
    one failed call ever rather than one per block during playback.
    """
    if state.channels == 1:
        try:
            return _call(plugin, audio, sample_rate, reset)
        except ValueError as exc:
            wanted = required_input_channels(exc)
            if audio.shape[0] != 1 or wanted is None or wanted < 2:
                raise
            state.channels = wanted
    out = _call(plugin, numpy.repeat(audio, state.channels, axis=0),
                sample_rate, reset)
    if out.shape[0] == 1:
        return out
    return out.mean(axis=0, keepdims=True)
