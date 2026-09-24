"""
The master bus: one effects chain over the SUMMED mix, after every track's own
chain and before the -1 dBTP safety limiter (player.py, audio_export.py).

Uses a stand-in plugin so nothing depends on which VST3s a machine has.
"""
import json
import types
import wave

import numpy as np
import pytest

import audio_export
import player
import project
import vst_host
from app_actions import ActionsMixin


class Stub:
    """Stands in for a pedalboard plugin: scales, or clips, and records the
    `reset` flag of every call."""

    def __init__(self, gain=0.5, clip=None):
        self.gain, self.clip = gain, clip
        self.resets = []

    def __call__(self, audio, sample_rate, reset=False):
        self.resets.append(reset)
        out = audio * self.gain
        return np.clip(out, -self.clip, self.clip) if self.clip else out


def chain_with(plugin):
    chain = vst_host.TrackChain()
    chain.slots.append(vst_host.PluginSlot("Stub", "stub.vst3", plugin))
    return chain


def make_player(master=None, tracks=1, keep=None):
    src = np.random.default_rng(1).integers(
        -8000, 8000, player.SAMPLE_RATE * 3, dtype=np.int16)
    p = player.Player()
    p.tracks = []
    for i in range(tracks):
        t = player.Track(f"spk{i}", src, "s.wav")
        t.chain = vst_host.TrackChain()
        p.tracks.append(t)
    p.duration = 3.0
    p.master_chain = master
    p.keep_ranges = keep or []
    p.edited_mode = bool(keep)
    return p


TIME = types.SimpleNamespace(outputBufferDacTime=0.0, currentTime=0.0)


def pull(p, frames=1024):
    out = np.zeros((frames, 1), np.float32)
    p._callback(out, frames, TIME, None)
    return out[:, 0]


# ------------------------------------------------------------------ playback

def test_master_is_applied_to_the_mix_in_playback():
    plain = pull(make_player())
    mastered = pull(make_player(master=chain_with(Stub(0.5))))
    assert plain.any()
    np.testing.assert_allclose(mastered, plain * 0.5, rtol=1e-6)


@pytest.mark.parametrize("how", ["none", "empty", "disabled", "bypassed"])
def test_no_master_processing_when_absent_off_or_bypassed(how):
    chain = None
    if how != "none":
        chain = chain_with(Stub(0.5)) if how != "empty" else vst_host.TrackChain()
    if how == "disabled":
        chain.enabled = False
    if how == "bypassed":
        chain.slots[0].bypassed = True
    np.testing.assert_array_equal(pull(make_player(master=chain)),
                                  pull(make_player()))


def test_master_sees_the_sum_not_each_track():
    """A hard clip only differs between clip(a+b) and clip(a)+clip(b)."""
    plain = pull(make_player(tracks=2))
    got = pull(make_player(master=chain_with(Stub(1.0, clip=0.1)), tracks=2))
    np.testing.assert_allclose(got, np.clip(plain, -0.1, 0.1), rtol=1e-6)
    assert not np.allclose(got, 2 * np.clip(plain / 2, -0.1, 0.1))


def test_master_restarts_on_seek_but_not_on_a_cut_jump():
    stub = Stub()
    # Two kept regions with a cut between them: the third block jumps the cut.
    p = make_player(master=chain_with(stub), keep=[(0.0, 0.05), (1.0, 1.5)])
    for _ in range(4):
        pull(p)
    assert stub.resets == [True, False, False, False]   # jumped, no restart

    p.seek(0.5)
    pull(p)
    assert stub.resets[-1] is True                      # an explicit seek


def test_master_restart_is_kept_if_the_block_was_declined(monkeypatch):
    """A block declined for a pending plugin load must not eat the restart."""
    stub = Stub()
    p = make_player(master=chain_with(stub))

    def declined(*args, **kwargs):
        raise vst_host.GateUnavailable("load pending")
    monkeypatch.setattr(p.master_chain, "process_slots", declined)
    pull(p)
    assert p._master_reset_pending is True


def test_a_failing_master_never_stops_the_stream():
    def broken(audio, sample_rate, reset=False):
        raise RuntimeError("plugin exploded")
    plain = pull(make_player())
    got = pull(make_player(master=chain_with(broken)))
    np.testing.assert_array_equal(got, plain)           # passed through


# -------------------------------------------------------------------- export

@pytest.fixture
def offline(monkeypatch):
    """apply_master renders a detached snapshot; a stub plugin cannot be
    reloaded from a path, so let the snapshot be the chain itself."""
    monkeypatch.setattr(vst_host.TrackChain, "snapshot",
                        lambda self, log=None: self)


def mix_of(n=player.SAMPLE_RATE):
    return (0.3 * np.sin(np.arange(n) / 20.0)).astype(np.float32)


def test_apply_master_processes_the_mix(offline):
    mix = mix_of()
    out = audio_export.apply_master(mix, chain_with(Stub(0.5)))
    np.testing.assert_allclose(out, mix * 0.5, rtol=1e-6)


def test_apply_master_leaves_the_mix_alone_without_a_master(offline):
    mix = mix_of()
    empty = vst_host.TrackChain()
    assert audio_export.apply_master(mix, None) is mix
    assert audio_export.apply_master(mix, empty) is mix
    off = chain_with(Stub(0.5))
    off.enabled = False
    assert audio_export.apply_master(mix, off) is mix


def test_apply_master_honours_cancel(offline):
    mix = mix_of()
    out = audio_export.apply_master(mix, chain_with(Stub(0.5)),
                                    should_cancel=lambda: True)
    assert out is mix


def test_export_puts_the_master_on_the_mixdown_but_not_on_stems(
        offline, monkeypatch, tmp_path):
    track = mix_of()

    def fake_render(paths, keep, mutes=None, chains=None, gains=None,
                    progress=None, want_audio=False, on_track=None,
                    should_cancel=None):
        if on_track:
            on_track(paths[0], track.copy())
        return track.copy(), []
    monkeypatch.setattr(audio_export, "render_tracks", fake_render)

    out = str(tmp_path / "ep.wav")
    written, _peak = audio_export.export_audio(
        out, [str(tmp_path / "a.wav")], [(0.0, 1.0)], stems=True,
        master_chain=chain_with(Stub(0.5)))

    def read(path):
        with wave.open(path, "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()),
                                 np.int16).astype(np.float32) / 32767.0
    mixdown = read(written[0])
    stem = read(written[1])
    np.testing.assert_allclose(mixdown, track * 0.5, atol=2e-4)
    np.testing.assert_allclose(stem, track, atol=2e-4)       # untouched


# --------------------------------------------------------------- project file

def _app(master):
    var = lambda v: types.SimpleNamespace(get=lambda: v)       # noqa: E731
    return types.SimpleNamespace(
        speaker_paths=[], aggressiveness=var(50), auto_mute_on=var(False),
        edits=[], mute_edits=[], player=types.SimpleNamespace(tracks=[]),
        track_chains=[], master_chain=master, intro_path=None,
        outro_path=None, export_stems=var(False), bake_effects=var(False))


def _restore(data):
    holder = types.SimpleNamespace(
        _pending_project=data, player=types.SimpleNamespace(tracks=[]),
        track_chains=[], master_chain=None, log=lambda m: None,
        _build_mixer=lambda: None)
    ActionsMixin._restore_project_audio_state(holder)
    return holder


def test_master_chain_survives_save_and_load():
    master = vst_host.TrackChain()
    master.add_native("gain", {"gain_db": -4.5})
    data = json.loads(json.dumps(project.build(_app(master))))

    holder = _restore(data)
    assert len(holder.master_chain.slots) == 1
    assert holder.master_chain.slots[0].key == "gain"
    assert holder.master_chain.slots[0].params["gain_db"] == -4.5
    assert holder.player.master_chain is holder.master_chain


def test_project_saved_before_the_master_bus_opens_with_an_empty_master():
    data = json.loads(json.dumps(project.build(_app(vst_host.TrackChain()))))
    del data["master_chain"]                    # what an old file looks like

    holder = _restore(data)
    assert holder.master_chain.slots == []
    assert holder.master_chain.enabled
    assert holder.player.master_chain is holder.master_chain


# ------------------------------------------------------- master plugin editor

def test_open_master_editor_is_fed_the_premaster_mix():
    """The editor's own plugin instance only reports GUI changes (bypass
    switches, knobs) and animates its meters once it is being fed audio - so
    the master chain must publish the mix to it, as a track chain does."""
    import queue
    master = chain_with(Stub(0.5))
    slot = master.slots[0]
    slot.editor_audio_queue = queue.Queue(maxsize=2)

    p = make_player(master=master)
    heard = pull(p)                                  # what is played: mastered
    fed = slot.editor_audio_queue.get_nowait()       # what the editor is given

    plain = pull(make_player())                      # the unmastered mix
    np.testing.assert_allclose(fed, plain, rtol=1e-6)
    np.testing.assert_allclose(heard, plain * 0.5, rtol=1e-6)


def test_master_editor_is_fed_even_when_the_master_is_off_or_bypassed():
    import queue
    master = chain_with(Stub(0.5))
    master.enabled = False
    slot = master.slots[0]
    slot.editor_audio_queue = queue.Queue(maxsize=2)
    pull(make_player(master=master))
    assert not slot.editor_audio_queue.empty()


def test_a_full_editor_queue_never_blocks_the_master():
    import queue
    master = chain_with(Stub(0.5))
    slot = master.slots[0]
    slot.editor_audio_queue = queue.Queue(maxsize=1)
    slot.editor_audio_queue.put(np.zeros(4, np.float32))     # already full
    plain = pull(make_player())
    np.testing.assert_allclose(pull(make_player(master=master)), plain * 0.5,
                               rtol=1e-6)
