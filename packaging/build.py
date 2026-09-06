#!/usr/bin/env python
"""
Builds the Windows installer, start to finish.

    python packaging/build.py

Steps, in order:
    1. icon        PNG -> autocut.ico + auto_cut/assets/autocut.png
    2. plugins     download the bundled VST3s (skipped if already there)
    3. PyInstaller dist/AutoCut/
    4. Inno Setup  packaging/output/Wavefield-Setup-<version>.exe

The version comes from auto_cut/version.py and is passed to Inno on the command
line, so the installer, the About box and the update check can no longer
disagree - previously autocut.iss hardcoded it and had a half-written attempt
at reading version.py that never ran.

Options:
    --skip-plugins   leave packaging/vst3 alone
    --no-installer   stop after PyInstaller
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGING = os.path.join(BASE, "packaging")

# Inno Setup installs per-user by default, not into Program Files, which is
# where everyone looks for it first.
ISCC_CANDIDATES = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""),
                 "Programs", "Inno Setup 6", "ISCC.exe"),
    r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 6\ISCC.exe",
]

# ffmpeg must be an LGPL build. The common "full" builds are GPL, which would
# put stronger obligations on the installer for codecs Wavefield never uses.
FFMPEG_DIR = os.environ.get(
    "AUTOCUT_FFMPEG_DIR",
    os.path.join(os.path.expanduser("~"), "ffmpeg-lgpl", "bin"))


def read_version():
    text = open(os.path.join(BASE, "auto_cut", "version.py"),
                encoding="utf-8").read()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("could not find __version__ in auto_cut/version.py")
    return match.group(1)


def find_iscc():
    for path in ISCC_CANDIDATES:
        if path and os.path.exists(path):
            return path
    return shutil.which("ISCC")


def step(number, title):
    print(f"\n=== {number}. {title} " + "=" * max(0, 50 - len(title)), flush=True)


def main():
    skip_plugins = "--skip-plugins" in sys.argv
    no_installer = "--no-installer" in sys.argv
    version = read_version()
    print(f"Wavefield {version}")

    step(1, "icon")
    sys.path.insert(0, PACKAGING)
    import make_icon
    make_icon.make()

    step(2, "bundled plugins")
    vst3 = os.path.join(PACKAGING, "vst3")
    if skip_plugins:
        print("  skipped (--skip-plugins)")
    elif os.path.isdir(vst3) and os.listdir(vst3):
        print(f"  already present: {vst3}")
    else:
        import fetch_plugins
        fetch_plugins.main()

    step(3, "PyInstaller")
    env = dict(os.environ, AUTOCUT_FFMPEG_DIR=FFMPEG_DIR,
               PYTHONIOENCODING="utf-8")
    if not os.path.isdir(FFMPEG_DIR):
        print(f"  WARNING: {FFMPEG_DIR} not found - the build will expect "
              f"ffmpeg on the user's PATH. Set AUTOCUT_FFMPEG_DIR.")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller",
         os.path.join(PACKAGING, "autocut.spec"),
         "--noconfirm",
         "--distpath", os.path.join(BASE, "dist"),
         "--workpath", os.path.join(BASE, "build")],
        cwd=BASE, env=env, check=True)

    if no_installer:
        print("\nstopped before the installer (--no-installer)")
        return

    step(4, "Inno Setup")
    iscc = find_iscc()
    if not iscc:
        raise SystemExit(
            "Inno Setup not found. Install it with:\n"
            "    winget install JRSoftware.InnoSetup")
    subprocess.run(
        [iscc, f"/DAppVersion={version}", os.path.join(PACKAGING, "autocut.iss")],
        cwd=PACKAGING, check=True)

    output = os.path.join(PACKAGING, "output",
                          f"Wavefield-Setup-{version}.exe")
    if os.path.exists(output):
        size = os.path.getsize(output) / 1e6
        print(f"\ndone: {output}  ({size:.0f} MB)")

        # Goes in the release notes. Until the builds are code-signed this is
        # the only way someone can confirm the installer they downloaded is
        # the one published here, so it is printed rather than left to be
        # remembered. Read in chunks: the installer is ~100 MB.
        digest = hashlib.sha256()
        with open(output, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        print(f"SHA-256: {digest.hexdigest()}")
        print("Paste that into the GitHub release notes - see docs/RELEASING.md")


if __name__ == "__main__":
    main()
