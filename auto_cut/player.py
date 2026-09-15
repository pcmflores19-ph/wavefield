"""
Multi-track audio playback for the editor.

Each speaker is decoded once to a raw mono PCM file and memory-mapped, so
seeking anywhere is instant and a 58-minute episode costs almost no RAM (the
OS page cache does the work). Playback mixes the tracks live with per-track
gain/mute/solo, running each track's effect chain live on every block - like
OBS Studio's filter chain on a stream. There is no pre-rendered/baked
alternative: rendering happens only at export (audio_export.py), which runs
the same chain in one continuous pass over the whole recording.

In EDITED mode the player skips the cut regions as it goes, so what you hear is
the exported timeline rather than the raw recording.
"""

import hashlib
import os
import queue
import subprocess
import threading

import numpy as np
import sounddevice as sd

import bundled
import effects
import settings

# Matches audio_export.MUTE_FADE_SECONDS - what you hear must be what you get.
MUTE_FADE_SECONDS = 0.010

# 48kHz to match camera audio and the audioRate FCPXML export declares -
# a baked export at 44.1kHz made Resolve unable to relink the media.
SAMPLE_RATE = 48000
CACHE_DIR = settings.cache_dir()


def _pcm_cache_path(audio_path):
    stat = os.stat(audio_path)
    key = hashlib.sha1(
        f"{audio_path}|{stat.st_size}|{stat.st_mtime}|{SAMPLE_RATE}".encode("utf-8")
    ).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, key + f".{SAMPLE_RATE}.mono.pcm")


def decode_to_pcm(audio_path):
    """Decodes to raw mono s16le at SAMPLE_RATE, cached. Returns the file path."""
    out_path = _pcm_cache_path(audio_path)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    tmp_path = out_path + ".part"
    cmd = [
        bundled.tool("ffmpeg"), "-v", "error", "-y", "-i", audio_path,
        "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg PCM decode failed for {os.path.basename(audio_path)}:\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    os.replace(tmp_path, out_path)
    return out_path


def decoded_duration_seconds(audio_path):
    """
    The REAL length of `audio_path`'s audio, measured from the same decoded
    PCM that playback and waveform peaks are generated from - not ffprobe's
    container/format duration (media_probe.probe()'s duration_seconds),
    which can disagree with the true decoded sample count (encoder priming/
    padding, VFR video, a probe index that doesn't match the real stream).

    Using ffprobe's duration to size/scale the waveform while playback runs
    on the decoded length is what caused waveforms to appear shifted left/
    right (growing the further into the track) or flatter than the real
    audio - confirmed by direct tracing 2026-09-13. Callers that need the
    real audio timeline (peak generation, waveform pixel scaling) should use
    this instead; callers that need the container/video's own duration
    (export, sync, FCPXML) should keep using media_probe's value.
    """
    samples = np.memmap(decode_to_pcm(audio_path), dtype=np.int16, mode="r")
    return samples.shape[0] / SAMPLE_RATE


class Track:
    def __init__(self, name, samples, path=None):
        self.name = name
        self.samples = samples      # np.memmap of int16
        self.path = path            # the source recording, for cache keys
        self.gain = 1.0
        self.muted = False          # whole-track mute (mixer button)
        self.soloed = False
        self.mute_ranges = []       # [(start_s, end_s)] hand-muted regions
        self.chain = None           # vst_host.TrackChain, run live every block
        # Levels of what this track actually contributes to the mix - measured
        # post-VST, post-mute, post-fader. Read by the UI meters.
        self.peak_level = 0.0
        self.rms_level = 0.0


class Player:
    def __init__(self, on_finished=None):
        self.tracks = []
        self.keep_ranges = []       # [(start_s, end_s)] - honoured in edited mode
        self.edited_mode = True
        self.duration = 0.0
        self.on_finished = on_finished

        self._pos = 0               # playhead, in samples, in SOURCE time
        self._seg = 0               # index into keep_ranges, for sequential playback
        # Level of the summed output, for the master meter.
        self.master_peak = 0.0
        self.master_rms = 0.0
        self._stream = None
        self._lock = threading.Lock()
        # Set on any position discontinuity (an explicit seek, or the
        # callback loop jumping over a cut) - the live effect chain now
        # runs continuously (see _mix_into), so this is what tells it "the
        # next block is not a continuation of the last one, start each
        # slot's internal state fresh" instead of carrying envelope/gate/
        # filter memory across a jump in time that memory was never meant
        # to span. Consumed (and cleared) by the very next _mix_into call.
        self._reset_pending = True

    # ---------- setup ----------

    def load(self, paths, names=None):
        self.stop()
        tracks = []
        for i, path in enumerate(paths):
            pcm_path = decode_to_pcm(path)
            samples = np.memmap(pcm_path, dtype=np.int16, mode="r")
            name = (names[i] if names else os.path.splitext(os.path.basename(path))[0])
            tracks.append(Track(name, samples, path))
        self.tracks = tracks
        self.duration = max((t.samples.size for t in tracks), default=0) / SAMPLE_RATE
        self._pos = 0
        self._seg = 0

    def set_keep_ranges(self, keep_ranges):
        with self._lock:
            self.keep_ranges = list(keep_ranges)
            self._resync_segment()

    # ---------- transport ----------

    @property
    def is_playing(self):
        return self._stream is not None and self._stream.active

    def play(self):
        if self.is_playing or not self.tracks:
            return
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=1024, callback=self._callback,
        )
        self._stream.start()

    def silence_levels(self):
        """Zeroes every meter reading - used when playback stops."""
        self.master_peak = 0.0
        self.master_rms = 0.0
        for t in self.tracks:
            t.peak_level = 0.0
            t.rms_level = 0.0

    def pause(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.silence_levels()

    def toggle(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def stop(self):
        self.pause()
        self.seek(0.0)

    def seek(self, seconds):
        with self._lock:
            self._pos = int(max(0.0, min(seconds, self.duration)) * SAMPLE_RATE)
            self._resync_segment()
            self._reset_pending = True

    def skip(self, delta_seconds):
        self.seek(self.position + delta_seconds)

    @property
    def position(self):
        return self._pos / SAMPLE_RATE

    # ---------- mixing ----------

    def _resync_segment(self):
        """Point _seg at the keep range containing or following the playhead."""
        if not (self.edited_mode and self.keep_ranges):
            self._seg = 0
            return
        t = self._pos / SAMPLE_RATE
        self._seg = 0
        for i, (_, end) in enumerate(self.keep_ranges):
            if end > t:
                self._seg = i
                break
        else:
            self._seg = len(self.keep_ranges)

    def _active_tracks(self):
        soloed = [t for t in self.tracks if t.soloed]
        pool = soloed if soloed else self.tracks
        return [t for t in pool if not t.muted]

    def _mix_into(self, out, start_sample, count, tracks, reset=False):
        end = start_sample + count
        for t in tracks:
            source = t.samples
            n = source.size
            if start_sample >= n:
                continue
            chunk = source[start_sample:min(end, n)]
            if not chunk.size:
                continue
            buf = chunk.astype(np.float32) * (1.0 / 32768.0)

            # Feed this track's raw pre-chain audio to any open plugin editor
            # in its chain, so that editor's own plugin instance (see
            # vst_host.open_editor_subprocess) has real signal to process -
            # otherwise a plugin GUI with a live meter (a de-esser's
            # gain-reduction display, say) never has anything to draw. Never
            # blocks: put_nowait drops the block if the editor's forwarder
            # thread is behind, since a dropped block only costs the meter a
            # moment of staleness, never the audio itself. Runs even if the
            # chain is disabled/bypassed - the tap is upstream of that check,
            # deliberately: the editor always reflects this track's own
            # signal, independent of whether its chain is currently applied.
            if t.chain is not None:
                for slot in t.chain.slots:
                    q = getattr(slot, "editor_audio_queue", None)
                    if q is not None:
                        try:
                            q.put_nowait(buf.copy())
                        except queue.Full:
                            pass

            # The whole chain, live, every block - like OBS's filter chain
            # on a stream. There is no pre-rendered/baked alternative any
            # more; rendering happens only at export (audio_export.py),
            # which runs the same chain in one continuous pass over the
            # whole recording. `reset` is True only on the
            # first block after a position discontinuity (an explicit seek,
            # or the callback loop jumping over a cut below) - every other
            # block continues the same slots' internal envelope/gate/filter
            # state instead of restarting it, so a continuously playing
            # dynamics effect sounds the same as one continuous pass (i.e.
            # the same as export), not like it resets every ~21ms.
            if t.chain is not None and t.chain.enabled and t.chain.slots:
                try:
                    # Never wait in the realtime callback: the player lock is
                    # held for the whole callback, so waiting here would also
                    # block pause, stop, and seek on the UI thread.
                    processed = t.chain.process_slots(
                        buf, SAMPLE_RATE, t.chain.slots, reset=reset,
                        gate_timeout=0.0)
                    if processed.size == buf.size:
                        buf = processed
                except Exception:
                    pass    # never let a plugin fault kill the audio stream

            buf = buf * t.gain

            # Silence any muted region overlapping this block, ramping at the
            # edges - a hard step to zero clicks, and auto-mute makes hundreds
            # of these.
            fade = max(1, int(MUTE_FADE_SECONDS * SAMPLE_RATE))
            for m_start, m_end in t.mute_ranges:
                a = int(m_start * SAMPLE_RATE) - start_sample
                b = int(m_end * SAMPLE_RATE) - start_sample
                if b <= 0 or a >= buf.size:
                    continue
                span = min(fade, max(1, (b - a) // 2))
                # Each edge is clipped to this block, so a mute spanning
                # several blocks still ramps only where the edge really is.
                fs, fe = a, min(a + span, buf.size)
                if fe > max(0, fs):
                    lo, hi = max(0, fs), fe
                    ramp = np.linspace(1.0, 0.0, span, dtype=np.float32)
                    buf[lo:hi] *= ramp[lo - fs:hi - fs]
                body_a, body_b = max(0, a + span), min(buf.size, b - span)
                if body_b > body_a:
                    buf[body_a:body_b] = 0.0
                rs, re = max(b - span, 0), b
                if re > rs and rs < buf.size:
                    lo, hi = max(0, rs), min(buf.size, re)
                    ramp = np.linspace(0.0, 1.0, span, dtype=np.float32)
                    buf[lo:hi] *= ramp[lo - rs:hi - rs]

            t.peak_level = effects.true_peak(buf)
            t.rms_level = float(np.sqrt(np.mean(buf * buf))) if buf.size else 0.0

            out[:buf.size] += buf

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            tracks = self._active_tracks()
            filled = 0
            while filled < frames:
                if self.edited_mode and self.keep_ranges:
                    if self._seg >= len(self.keep_ranges):
                        break
                    seg_start, seg_end = self.keep_ranges[self._seg]
                    seg_start_s = int(seg_start * SAMPLE_RATE)
                    seg_end_s = int(seg_end * SAMPLE_RATE)
                    if self._pos < seg_start_s:
                        self._pos = seg_start_s          # jump the cut
                        self._reset_pending = True
                    if self._pos >= seg_end_s:
                        self._seg += 1
                        continue
                    take = min(frames - filled, seg_end_s - self._pos)
                else:
                    total = int(self.duration * SAMPLE_RATE)
                    if self._pos >= total:
                        break
                    take = min(frames - filled, total - self._pos)

                reset = self._reset_pending
                self._reset_pending = False
                self._mix_into(out[filled:filled + take], self._pos, take,
                               tracks, reset=reset)
                self._pos += take
                filled += take

        # Master level measured before limiting (true peak, so overs the
        # limiter is about to catch - and overs a lossy export re-encode
        # could add on top - are both visible), so overs are visible.
        self.master_peak = effects.true_peak(out)
        self.master_rms = float(np.sqrt(np.mean(out * out))) if out.size else 0.0

        ceiling = 10 ** (effects.LIMITER_CEILING_DB / 20.0)
        if self.master_peak > ceiling:
            # Limit the loud moment instead of hard-clipping it, so what you
            # hear live matches what the export does to the same overlap.
            out = effects.limiter(out, SAMPLE_RATE, threshold_db=effects.LIMITER_CEILING_DB)
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:, 0] = out

        if filled == 0:
            self.silence_levels()
            raise sd.CallbackStop()

    def close(self):
        self.pause()
        self.tracks = []
