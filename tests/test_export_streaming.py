"""
Export memory streaming.

export_audio used to hold every speaker's fully rendered track in a list
before mixing (~5GB for four 2-hour speakers, before the mix buffer existed),
and write_wav built four whole-track temporaries. Both now stream.

The plugin call itself was deliberately left alone, so the exported audio must
not move by a single byte - that is what these tests check, against a
reimplementation of the original.
"""

import hashlib
import os
import wave

import numpy as np
import pytest

import audio_export
import effects


# --------------------------------------------------------------- the original

def reference_load(path_map, name):
    mapped = np.memmap(path_map[name], dtype=np.int16, mode="r")
    return np.asarray(mapped, dtype=np.float32) / 32768.0


def reference_render(path_map, name, keeps, mutes, chain, gain):
    audio = reference_load(path_map, name)
    if chain is not None and chain.active_slots():
        processed = chain.snapshot().process(audio, 48000, reset=True)
        if processed.size == audio.size:
            audio = processed
    fade = max(1, int(0.010 * 48000))
    for start, end in (mutes or []):
        a = max(0, int(start * 48000))
        b = min(audio.size, int(end * 48000))
        if b <= a:
            continue
        span = min(fade, (b - a) // 2) or 1
        audio[a:a + span] *= np.linspace(1.0, 0.0, span, dtype=np.float32)
        if b - span > a + span:
            audio[a + span:b - span] = 0.0
        audio[b - span:b] *= np.linspace(0.0, 1.0, span, dtype=np.float32)
    if gain != 1.0:
        audio *= gain
    pieces = []
    for start, end in keeps:
        a = max(0, int(start * 48000))
        b = min(audio.size, int(end * 48000))
        if b > a:
            pieces.append(audio[a:b])
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def reference_write(path, audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(pcm.tobytes())
    return path


def reference_export(path_map, out_path, names, keeps, mutes, chains, gains,
                     stems, intro=None, outro=None):
    rendered = [
        reference_render(path_map, name, keeps,
                         [(s, e) for lane, s, e in mutes if lane == i],
                         chains[i] if chains else None,
                         gains[i] if gains else 1.0)
        for i, name in enumerate(names)
    ]
    length = max(a.size for a in rendered)
    mix = np.zeros(length, dtype=np.float32)
    for audio in rendered:
        mix[:audio.size] += audio

    peak = float(np.abs(mix).max()) if mix.size else 0.0
    if peak > 1.0:
        mix = effects.limiter(mix, 48000, threshold_db=0.0)

    bookends = []
    if intro is not None:
        bookends.append(intro)
    bookends.append(mix)
    if outro is not None:
        bookends.append(outro)
    if len(bookends) > 1:
        mix = np.concatenate(bookends)

    written = [reference_write(out_path, mix)]
    if stems:
        base, ext = os.path.splitext(out_path)
        for i, audio in enumerate(rendered):
            written.append(
                reference_write(f"{base}_{names[i]}{ext or '.wav'}", audio))
    return written, peak


# ------------------------------------------------------------------- fixtures

@pytest.fixture
def speakers(tmp_path, monkeypatch):
    """Three int16 PCM tracks of different lengths, wired into audio_export."""
    rng = np.random.default_rng(7)
    path_map = {}
    names = []
    for index, seconds in enumerate([9.0, 11.5, 7.25]):
        name = f"spk{index}"
        samples = rng.integers(-25000, 25000, int(48000 * seconds),
                               dtype=np.int16)
        path = tmp_path / f"{name}.pcm"
        path.write_bytes(samples.tobytes())
        path_map[name] = str(path)
        names.append(name)

    monkeypatch.setattr(audio_export, "decode_to_pcm", lambda n: path_map[n])
    return path_map, names


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


KEEPS = [(0.5, 4.0), (5.0, 9.0), (9.5, 11.0)]
MUTES = [(0, 1.0, 1.5), (1, 6.0, 6.2), (2, 2.0, 2.4)]
GAINS = [1.0, 0.8, 1.3]


@pytest.mark.parametrize("stems", [False, True])
def test_export_is_byte_identical(speakers, tmp_path, stems):
    path_map, names = speakers

    want, want_peak = reference_export(
        path_map, str(tmp_path / "ref.wav"), names, KEEPS, MUTES, None,
        GAINS, stems)
    got, got_peak = audio_export.export_audio(
        str(tmp_path / "new.wav"), names, KEEPS, mutes=MUTES, chains=None,
        gains=GAINS, stems=stems)

    assert got_peak == want_peak
    assert len(got) == len(want)
    # Order matters: callers show this list, mixdown first then stems.
    for want_path, got_path in zip(want, got):
        assert os.path.basename(want_path).replace("ref", "") == \
               os.path.basename(got_path).replace("new", "")
        assert digest(want_path) == digest(got_path), got_path


def test_export_is_byte_identical_with_a_chain(speakers, tmp_path):
    import vst_host

    path_map, names = speakers
    chain = vst_host.TrackChain()
    chain.add_native("compressor")
    chains = [chain, None, None]

    want, want_peak = reference_export(
        path_map, str(tmp_path / "ref.wav"), names, KEEPS, MUTES, chains,
        GAINS, False)
    got, got_peak = audio_export.export_audio(
        str(tmp_path / "new.wav"), names, KEEPS, mutes=MUTES, chains=chains,
        gains=GAINS, stems=False)

    assert got_peak == want_peak
    assert digest(want[0]) == digest(got[0])


def test_export_is_byte_identical_with_bookends(speakers, tmp_path, monkeypatch):
    """The intro/outro beds are written as separate parts, not concatenated."""
    path_map, names = speakers
    rng = np.random.default_rng(11)
    intro = (rng.random(48000).astype(np.float32) - 0.5)
    outro = (rng.random(24000).astype(np.float32) - 0.5)

    beds = {"intro.wav": intro, "outro.wav": outro}
    monkeypatch.setattr(audio_export, "decode_audio_file",
                        lambda p: beds[os.path.basename(p)])

    want, _ = reference_export(
        path_map, str(tmp_path / "ref.wav"), names, KEEPS, MUTES, None,
        GAINS, False, intro=intro, outro=outro)
    got, _ = audio_export.export_audio(
        str(tmp_path / "new.wav"), names, KEEPS, mutes=MUTES, chains=None,
        gains=GAINS, stems=False,
        intro_path="intro.wav", outro_path="outro.wav")

    assert digest(want[0]) == digest(got[0])


def test_load_track_matches_the_two_array_version(speakers):
    path_map, names = speakers
    got = audio_export._load_track(names[0])
    mapped = np.memmap(path_map[names[0]], dtype=np.int16, mode="r")
    want = np.asarray(mapped, dtype=np.float32) / 32768.0
    # Scaling by 2**-15 is exact, so this is equality, not closeness.
    assert np.array_equal(got, want)


def test_write_wav_accepts_parts(tmp_path):
    rng = np.random.default_rng(13)
    first = (rng.random(1000).astype(np.float32) - 0.5)
    second = (rng.random(500).astype(np.float32) - 0.5)

    joined = audio_export.write_wav(str(tmp_path / "parts.wav"),
                                    [first, second])
    single = audio_export.write_wav(str(tmp_path / "single.wav"),
                                    np.concatenate([first, second]))

    assert digest(joined) == digest(single)


def test_mix_memory_does_not_scale_with_speaker_count(tmp_path, monkeypatch):
    """
    The point of summing incrementally: N speakers must not cost N tracks.

    Compared at four speakers against eight, both of which sit on the flat
    part of the curve - only the current speaker and the mix are ever alive
    together, so doubling the speakers must cost nothing. (One or two
    speakers measure lower simply because there is no second iteration, which
    would make them a misleading baseline.)
    """
    import tracemalloc

    rng = np.random.default_rng(17)
    seconds = 20.0

    def peak_for(speaker_count):
        path_map = {}
        names = []
        for index in range(speaker_count):
            name = f"s{speaker_count}_{index}"
            samples = rng.integers(-25000, 25000, int(48000 * seconds),
                                   dtype=np.int16)
            path = tmp_path / f"{name}.pcm"
            path.write_bytes(samples.tobytes())
            path_map[name] = str(path)
            names.append(name)
        monkeypatch.setattr(audio_export, "decode_to_pcm",
                            lambda n: path_map[n])

        tracemalloc.start()
        audio_export.export_audio(str(tmp_path / f"out{speaker_count}.wav"),
                                  names, [(0.0, seconds)])
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    four = peak_for(4)
    eight = peak_for(8)

    # Before, eight speakers held eight rendered tracks plus the mix.
    assert eight < four * 1.1, f"peak scaled with speakers: {four} -> {eight}"
