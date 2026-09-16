# Developer notes

Everything here is for people working on Wavefield's code. Users never need any
of it - they run the installer, which bundles Python, ffmpeg and the plugins.

## Running from source

You need **Python 3.10+** with tkinter (bundled on Windows and macOS; on
Debian/Ubuntu `sudo apt install python3-tk`), and **ffmpeg + ffprobe** on your
PATH.

```bash
git clone https://github.com/pcmflores19-ph/wavefield.git
cd wavefield
pip install -r requirements.txt
python auto_cut/app.py
```

Or `launch_wavefield.bat` on Windows, `./launch_wavefield.sh` elsewhere.

## Building the installer

See [RELEASING.md](RELEASING.md). Short version: `python packaging/build.py`.

## Transcription

Optional, and never bundled - WhisperX pulls in PyTorch and CUDA, several
gigabytes for a feature most users skip.

```bash
pip install whisperx
```

Found on PATH, or point at it with `AUTOCUT_WHISPERX=/path/to/whisperx`.
Uses an NVIDIA GPU when present, falls back to CPU otherwise; force it with
`AUTOCUT_DEVICE=cpu` or `cuda`.

All 100 Whisper languages are offered. Whisper transcribes all of them, but
WhisperX only ships word-level *alignment* for about 40. Without alignment the
word timings are coarse - which now only affects how precisely the transcript
highlights during playback, since the cuts come from the waveform.

## How the cut logic works

Per track, for analysis only:

1. **Normalise.** Recordings come from different rooms, mics and calls, so one
   person's silence can sit louder than another's speech. The level is taken
   from the 95th percentile of frame energy rather than overall RMS — most of a
   podcast track is one person *not* talking, so overall RMS mostly measures how
   quiet their room is, and normalising by it would amplify the quietest
   recording the most.
2. **Denoise** with rnnoise. Room tone and fan noise are what a plain energy
   gate mistakes for talking.
3. **Gate** with hysteresis, then merge speech separated by less than 0.15s.

That merge matters more than any threshold. Without it the gate flickers, and a
single 20ms blip in the middle of a long silence splits it into two halves each
too short to cut.

Detected regions are then widened by 0.10s at each end, and a further 0.25s is
kept around every cut. A word does not start at full volume — "s", "f" and soft
first syllables cross the gate slightly after the sound began — so cutting at
the detected boundary clips the first or last letter.

Finally, a stretch is removed only if **every** speaker is silent through it.

---

## FCPXML structure (why it looks the way it does)

Resolve's FCPXML importer decides the track layout itself, using heuristics you
cannot control from the file. Getting two clean tracks took a series of
hand-written import tests. What they showed:

- Speaker 0 goes in the **primary storyline**; everyone else is a **connected
  clip** on lane N. That yields V1/A1, V2/A2 and nothing else. Hanging every
  speaker off a base `<gap>` instead also works, but the gap itself imports as
  an empty V1.
- Each speaker needs a **unique audio role** (`dialogue.<name>`). With one
  shared `dialogue` role, Resolve loses track of whose audio is whose once the
  timeline fragments into many cut segments, and scatters it across tracks.
- Mono sources must be **declared** mono. Claiming `audioChannels="2"` for a
  mono file caused the same scattering.
- Host and guest are split on the **union** of both their mute boundaries, so
  their pieces line up one to one and each guest piece can hang off the host
  piece it sits over. A connected clip's `offset` is measured in its parent
  clip's local time, not timeline time.

Camera switching was attempted and removed. Doing it properly needs Resolve's
scripting API to enable and disable clips on a timeline it already owns —
exactly what the free edition blocks. Every attempt to express it in FCPXML
added clips that Resolve then redistributed on import.

---

## Layout

| File | Role |
|---|---|
| `auto_cut/app.py` | Application and entry point |
| `auto_cut/app_ui.py` | Widget construction |
| `auto_cut/app_actions.py` | Project I/O and transcript editing |
| `auto_cut/voice_activity.py` | Finds speech from the waveform |
| `auto_cut/silence_detector.py` | Merges speakers, computes keep ranges |
| `auto_cut/player.py` | Multitrack playback (memory-mapped PCM, live mixing) |
| `auto_cut/waveform.py` | Waveform peaks, with each speaker's VST chain applied |
| `auto_cut/vst_host.py` | VST3 discovery and per-track effect chains |
| `auto_cut/fcpxml_writer.py` | Writes the Resolve timeline |
| `auto_cut/audio_export.py` | Renders the finished WAV |
| `auto_cut/whisperx_runner.py` | Optional transcription |
| `auto_cut/project.py` | Save, open, autosave |

---

## Known limits

- **Speakers must start together.** All recordings are assumed to share a
  timeline origin, which is what OBS Source Record produces. There is no drift
  correction and no sync offset.
- Plugin editor windows are raised to the front only on Windows. The plugins
  themselves work everywhere.
- Video is never re-encoded, and there is no video preview — the waveform is
  the editing surface.
