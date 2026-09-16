"""
The text behind the Help menu.

Kept as data in its own module because the installed app has no README beside
it - for anyone who gets a built copy rather than the repository, this is the
only documentation there is.
"""

from version import APP_NAME, SOURCE_URL, __version__

QUICK_START = """\
WHAT THIS IS FOR

Recordings where every speaker was captured to their own file - OBS Source
Record, Riverside, local Zoom recordings. Wavefield finds where each person is
actually talking, removes the stretches where nobody is, and exports either a
DaVinci Resolve timeline or a finished WAV.


1. ADD YOUR RECORDINGS - HOST FIRST

The order decides which track each speaker lands on: the first file becomes
V1/A1, the second V2/A2, and so on. Use Up to reorder.

All the recordings should start at the same moment - that is what OBS
produces. If they do not, press Sync: it detects, per recording, how far off
it is from a reference you pick (by matching either when each person talks,
or the room tone itself if they were recorded together), and trims or pads a
synced copy for you to review before anything is changed for real.


2. WAIT A MOMENT

Nothing to click. Adding a recording starts the work: speech is found from the
waveform, with each track normalised, denoised and run through a gate that
decides talking from not-talking. Neither the normalising nor the denoising
touches your audio - they only inform the decision.

The first run on a file has to decode it, which takes a few minutes for an
hour-long recording. After that it is cached and re-runs are quick. The bar at
the bottom of the window shows how it is getting on.

Auto-cut and Auto-mute are the two buttons underneath. They light up when they
are on. Auto-cut starts on; Auto-mute does not, because it only makes sense
with a microphone each.

TRANSCRIBE is separate, and deliberately not automatic - it is minutes of work
even on a good graphics card, and plenty of episodes never need one. Pressing
it asks which language, and which model.

The model decides how good and how slow the transcript is. Bigger is better and
much slower: tiny and base suit any laptop, small is a fair compromise on a
processor, and the large models really want an NVIDIA graphics card. Choosing a
large model without one is the most common reason people think the app has
frozen - it has not, it is just going to take hours. Wavefield warns you in the
dialog when that is what you have picked.


3. SET THE AGGRESSIVENESS

The slider is the shortest pause that gets removed: 0 leaves anything under
three seconds alone, 100 cuts pauses as short as a quarter of a second. The
waveform re-shades as you drag, so you can see what each setting costs before
committing to it.

Turn Auto-cut dead air off if you would rather cut entirely by hand.


4. CHECK IT, AND FIX WHAT IS WRONG

Every speaker has their own waveform lane. Doomed stretches are shaded red.

  click            move the playhead
  drag             select a region
  shift + drag     pan
  mouse wheel      zoom

  q                delete the selection
  w                restore it
  a                mute this lane over the selection
  s                unmute it
  z                undo
  x                clear every hand edit

Monitor: Raw / Edited switches between hearing the original and hearing what
you are about to export - cuts, mutes, levels and effects included.

Auto-mute inactive speaker silences each microphone whenever its owner is not
talking, which removes bleed, breathing and keyboard noise from the idle mic.


5. EFFECTS (OPTIONAL)

Each track has its own effects chain, opened with its FX button. What you hear
updates as you move a slider.

Six effects are built in - Noise Gate, Compressor, Expander, Limiter, 3-Band EQ
and Gain - working the same way as the ones in OBS Studio, and starting from
the same settings. rnnoise is bundled as a plugin for noise suppression, and
any VST3 you install yourself appears alongside them.

A sensible starting order for a voice is: rnnoise, then Noise Gate, then
Compressor, then 3-Band EQ, with Limiter last to catch peaks.

Once a chain sounds right, save it with Presets at the bottom of the window and
load it again on the next episode instead of rebuilding it.

Effects are rendered into the WAV export. They are NOT written into the Resolve
timeline, which points at your untouched original recordings - do that side of
the work in Resolve.


6. EXPORT, FROM THE EXPORT MENU

  Timeline for DaVinci Resolve
      Then in Resolve: File > Import > Timeline > Import AAF, EDL, XML...
      You get two video tracks and two mono audio tracks.

  Finished audio (WAV)
      Cuts, mutes, effects and levels all rendered in. For an audio podcast
      this is the whole job - no round trip through Resolve.

  Finished video (MP4)
      A single rendered video file - the picture from the Vodcast tab's camera
      switching if it is set up, otherwise the first speaker's recording.
      Cuts, mutes, effects and levels are all rendered in, the same as the WAV.

Intro and outro audio, if you set them, are added to the WAV only, and are
never cut or processed.


7. VODCAST (OPTIONAL) - AUTOMATIC CAMERA SWITCHING

Only for a two-person episode filmed on two cameras, with a third recording
that already has both people in frame - see Vodcast > Read me for exactly
what is required and how it works.


SAVING

File > Save project keeps your files, edits, levels and effect chains in one
.wavefield_project file. The app also autosaves, and offers to recover after
a crash.
"""

SHORTCUTS = """\
EDITING                          TRANSPORT

  q    delete selection            space   play / pause
  w    restore selection
  a    mute lane in selection    PROJECT
  s    unmute lane
  z    undo last edit              Ctrl+N  new project
  x    clear all hand edits        Ctrl+O  open project
                                   Ctrl+S  save project

TIMELINE

  click             move the playhead
  drag              select a region
  shift + drag      pan across the timeline
  mouse wheel       zoom in and out
  double-click      (transcript) seek to that line
"""

TROUBLESHOOTING = """\
"ffmpeg is required" when starting
    Wavefield needs ffmpeg to read audio. The installed version ships with its
    own copy, so if you see this, try reinstalling.

No sound during playback
    Playback uses whatever your operating system has set as the default output
    device. Change it in your sound settings and restart Wavefield.

The FX window is empty
    The installed version ships its own plugins, so this should not happen -
    try reinstalling. Running from source, only VST3 plugins are found, and
    only in the standard folder for your system. VST2 is not supported.

Transcription never happens
    It needs WhisperX. The installer sets this up for you automatically -
    it is ticked by default on the last page of setup - so this should only
    happen if that box was unchecked. Fix it from File > Settings > Install
    WhisperX, no reinstall needed. Everything else works without it - the
    cuts do not depend on the transcript at all.

Transcription is very slow
    Without an NVIDIA graphics card it runs on the processor, which is slow for
    a long recording. Use a smaller model, or skip it.

Reading the recordings seems stuck
    The first pass on a file decodes the whole recording, which can take a few
    minutes per hour of audio. Watch the LOG panel - it reports each step.

The app closed by itself
    An audio plugin misbehaving can take the whole app down, and Python cannot
    catch that. It is recorded in autocut_crash.log next to the program.
    Sending that file with a bug report helps enormously.
"""

VODCAST_README = """\
WHAT THIS IS FOR

Automatic camera switching for a two-person video podcast filmed on separate
cameras - cutting between a wide shot of both people and single shots of
whoever is talking, without doing it by hand.


WHAT YOU NEED

  - Exactly two speaker recordings (host and guest), each WITH picture.
    Audio-only recordings have nothing to cut between.

  - A third recording (V3) that already has both people in frame - the wide
    or "two-shot". Set it with Vodcast > Set merged video (V3).

  - All three recordings must start at the same moment and run the same
    length. There is no sync correction: a file that starts late stays late,
    and a mismatch of more than half a second is rejected outright.

If any of this is missing, Vodcast > Switch cameras automatically stays
greyed out, and tells you what is missing when you try it anyway.


HOW IT WORKS

Turn on "Switch cameras automatically" and Wavefield decides, from who is
talking, when to show the host, the guest, or the two-shot (V3) - a cut away
to V3 happens automatically if one person is held on camera too long.

Shot length... sets the two limits that shape the result: the shortest a shot
may run (a brief reply should not cut away and back within a moment) and the
longest (nobody should be held on camera so long it looks stuck).

Camera changes are hand-edited the same way ordinary cuts are: drag along a
row in the CAMERAS strip to set which camera plays for a stretch. Hand edits
are sticky - they survive re-analysis and slider changes - so Vodcast >
Regenerate camera switching exists to throw them away and switch again from
the audio alone, if you would rather start over.


EXPORT

Camera switching carries into both picture exports: Export > Finished video
(MP4) renders it directly, and Export > Timeline for DaVinci Resolve writes
the camera changes as video-track edits on the V1/V2/V3 lanes, ready to
review or refine in Resolve. It has no effect on the WAV export, which is
audio only.
"""

ABOUT = f"""\
{APP_NAME} {__version__}

Removes dead air from multitrack podcast recordings.

Everything runs on your own machine. Nothing is uploaded, and there is no
account or subscription.

Free to use and to pass on.

Built on ffmpeg, numpy, sounddevice and pedalboard.
Transcription, when enabled, uses WhisperX.
Built-in effects (gate, compressor, expander, limiter, EQ, gain) are
Wavefield's own, modelled on the ones in OBS Studio.
Bundled plugin: rnnoise, for noise suppression.

This program includes GPL-licensed components, so it is distributed under
the GNU General Public License version 3. That means you are free to read,
change and share it. The complete source code is at

{SOURCE_URL}

See THIRD-PARTY-NOTICES.txt, installed alongside the program, for the
licences of everything included.
"""
