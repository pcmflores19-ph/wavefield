#!/usr/bin/env python
"""
Downloads the VST3 plugins that ship with a build, into packaging/vst3/.

Only open-source, redistributable plugins. "Free" on a plugin site nearly
always means free of charge, not free to redistribute - the freeware used
during development (TDR, Soap Voice Cleaner and friends) may not be put inside
an installer without written permission, so none of it is here.

  rnnoise      GPL-3.0   noise suppression. Not a luxury: voice_activity uses
                         it to denoise before gating, and speech detection is
                         measurably worse without it. Bundling it is what makes
                         the auto-cut behave the same on every machine.
  pp-track     GPL-3.0   PodcastPlugins TRACK (Klaus Scheuermann, built with
                         DPF): per-speaker leveling/EQ/dynamics. Unmodified
                         upstream binary. It only offers a stereo bus;
                         Wavefield's mono tracks reach it through
                         auto_cut/channel_adapt.py. The same project's MASTER
                         plugin is deliberately NOT bundled: it is a
                         master-bus finisher and Wavefield has no master bus.
ZamPlugins used to be bundled here as a voice chain. They are gone: the gate,
compressor, EQ, expander, limiter and gain are now built into the app
(auto_cut/effects.py, ported from OBS), which means plain sliders, nothing to
redistribute, and no third-party plugin GUI to host.

rnnoise stays because nothing replaces it - it is a trained noise-suppression
model, not a few lines of DSP.

pedalboard already makes the built application GPL-3, so this GPL plugin adds
no obligation that is not there already.

The binaries are NOT committed - packaging/vst3/ is gitignored. Run this once
before building, or let packaging/build.py do it.
"""

import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "packaging", "vst3")
LICENSE_DIR = os.path.join(OUT_DIR, "licences")

# Asset names were checked against the GitHub API rather than guessed - the
# first attempt at both of these 404'd.
RNNOISE_URL = ("https://github.com/werman/noise-suppression-for-voice/releases/"
               "download/v1.10/win-rnnoise.zip")

# PodcastPlugins release 1.0.0. The zip holds ready-made pp-track.vst3 and
# pp-master.vst3 bundles (plus .exe standalones we ignore); only pp-track is
# taken, see the module docstring for why not pp-master. The hash pins the
# exact file that was tested, so a re-tagged or tampered download fails the
# build instead of shipping. Source: https://github.com/trummerschlunk/PodcastPlugins
PODCASTPLUGINS_URL = ("https://github.com/trummerschlunk/PodcastPlugins/"
                      "releases/download/1.0.0/podcast-plugins-1.0.0-win64.zip")
PODCASTPLUGINS_SHA256 = (
    "774a53e6cd71cb994770bf07cc3a441c8bcdaaa84ea0375e8e66eb66e4560ba8")
PODCASTPLUGINS_WANTED = ("pp-track",)


def _download(url):
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "autocut-build"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def _extract_vst3(data, wanted=None, limit=None):
    """
    Pulls .vst3 files (or bundle directories) out of a zip into OUT_DIR.

    Returns the names taken. `wanted` filters by substring; without it,
    everything is taken.
    """
    taken = []
    archive = zipfile.ZipFile(io.BytesIO(data))
    for entry in archive.namelist():
        if entry.endswith("/"):
            continue
        lowered = entry.lower()
        if ".vst3" not in lowered:
            continue
        name = os.path.basename(entry.rstrip("/"))
        stem = name.lower().replace(".vst3", "")
        if wanted and stem not in wanted:
            continue
        if limit and len(taken) >= limit:
            break
        # Preserve a bundle's inner structure (Contents/<arch>/Name.vst3);
        # vst_host._resolve_binary relies on it.
        marker = lowered.find(".vst3")
        relative = entry[:marker + 5].split("/")[-1] + entry[marker + 5:]
        target = os.path.join(OUT_DIR, relative.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with archive.open(entry) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        taken.append(relative)
    return taken


def verify():
    """
    Loads every bundled plugin, so a broken one is caught here rather than as
    an empty FX window on someone else's machine.
    """
    sys.path.insert(0, os.path.join(BASE, "auto_cut"))
    import vst_host

    # Resolve OUT_DIR's own entries rather than going through
    # discover_plugins(): that de-duplicates by name with system folders
    # first, so a machine that already has rnnoise or the PodcastPlugins
    # installed would hide the copies actually being bundled here.
    bundled = []
    for entry in sorted(os.listdir(OUT_DIR)):
        if entry.lower().endswith(".vst3"):
            binary = vst_host._resolve_binary(os.path.join(OUT_DIR, entry))
            if binary:
                bundled.append((os.path.splitext(entry)[0], binary))
    if not bundled:
        print("  WARNING: nothing discoverable in packaging/vst3")
        return False

    ok = True
    import pedalboard
    for name, path in bundled:
        try:
            pedalboard.load_plugin(path)
            print(f"  ok    {name}")
        except Exception as exc:
            print(f"  FAIL  {name}: {str(exc)[:80]}")
            ok = False
    return ok


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LICENSE_DIR, exist_ok=True)

    print("rnnoise (GPL-3.0):")
    try:
        taken = _extract_vst3(_download(RNNOISE_URL))
        print(f"  took {len(taken)}: {', '.join(taken) or 'nothing'}")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    print("PodcastPlugins TRACK (GPL-3.0):")
    try:
        data = _download(PODCASTPLUGINS_URL)
        digest = hashlib.sha256(data).hexdigest()
        if digest != PODCASTPLUGINS_SHA256:
            raise RuntimeError(f"checksum mismatch (got {digest}) - refusing "
                               f"to bundle an unverified download")
        taken = _extract_vst3(data, wanted=PODCASTPLUGINS_WANTED)
        print(f"  took {len(taken)}: {', '.join(taken) or 'nothing'}")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    print("verifying every bundled plugin loads:")
    verify()
    print(f"\n{OUT_DIR}")


if __name__ == "__main__":
    main()
