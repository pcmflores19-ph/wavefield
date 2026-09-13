# PyInstaller spec for Auto-Cut.
#
# Build from the repository root:
#     pyinstaller packaging/autocut.spec --noconfirm
#
# Produces dist/AutoCut/ - a one-folder build. Deliberately not one-file:
# a single .exe unpacks itself to a temp directory on every launch, which is
# slow and is exactly the behaviour antivirus heuristics flag.
#
# ffmpeg is copied in from FFMPEG_DIR so the installed app has no external
# dependencies at all. WhisperX is NOT bundled and never will be - it pulls in
# torch and CUDA, several gigabytes, for a feature that is optional.

import os
import shutil

BASE = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(BASE, "auto_cut")

# Where to find ffmpeg.exe / ffprobe.exe to ship. Override with the
# AUTOCUT_FFMPEG_DIR environment variable.
FFMPEG_DIR = os.environ.get("AUTOCUT_FFMPEG_DIR", r"C:\ffmpeg\bin")


def _version_resource():
    """
    Writes the Windows version resource, and returns its path.

    An .exe carrying no version information at all - no product name, no
    company, no version - is a small antivirus heuristic in its own right, and
    it is also what Explorer shows under Properties. Generated from version.py
    so it cannot drift from the installer and the About box.

    This is not a substitute for signing. It removes one free reason to be
    suspicious of an unsigned binary, nothing more.
    """
    import re
    text = open(os.path.join(SRC, "version.py"), encoding="utf-8").read()
    version = re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)
    parts = (version.split(".") + ["0", "0", "0", "0"])[:4]
    quad = ", ".join(str(int(p)) for p in parts)

    resource = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(QUAD), prodvers=(QUAD),
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
    date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
        StringStruct('CompanyName', 'Behind The Science Podcast'),
        StringStruct('FileDescription', 'Wavefield - podcast post-production'),
        StringStruct('FileVersion', 'VERSION'),
        StringStruct('InternalName', 'Wavefield'),
        StringStruct('OriginalFilename', 'Wavefield.exe'),
        StringStruct('ProductName', 'Wavefield'),
        StringStruct('ProductVersion', 'VERSION'),
    ])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""".replace("QUAD", quad).replace("VERSION", version)

    path = os.path.join(SPECPATH, "version_info.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(resource)
    return path


def _ffmpeg_binaries():
    """(source, dest) pairs for the two ffmpeg tools, if we can find them."""
    found = []
    for name in ("ffmpeg", "ffprobe"):
        exe = name + (".exe" if os.name == "nt" else "")
        path = os.path.join(FFMPEG_DIR, exe)
        if not os.path.exists(path):
            path = shutil.which(name) or ""
        if path and os.path.exists(path):
            found.append((path, "ffmpeg"))
        else:
            print(f"WARNING: {exe} not found - the build will need ffmpeg on "
                  f"the user's PATH. Set AUTOCUT_FFMPEG_DIR to bundle it.")
    return found


def _bundled_plugins():
    """
    The open-source voice chain from fetch_plugins.py, if it has been run.

    Shipping rnnoise is not cosmetic: voice_activity uses it to denoise before
    gating, so without it the auto-cut is measurably worse on a machine that
    has no copy installed.
    """
    source = os.path.join(SPECPATH, "vst3")
    if not os.path.isdir(source):
        print("WARNING: packaging/vst3 is empty - run fetch_plugins.py to "
              "bundle the plugins, or the FX window will be empty on a "
              "clean machine.")
        return []
    return [(source, "vst3")]


a = Analysis(
    [os.path.join(SRC, "app.py")],
    pathex=[SRC],                  # flat sibling imports: `import ui_theme`
    binaries=_ffmpeg_binaries(),
    datas=[
        (os.path.join(SPECPATH, "THIRD-PARTY-NOTICES.txt"), "."),
        (os.path.join(SPECPATH, "FFMPEG-LICENSE.txt"), "."),
        (os.path.join(BASE, "LICENSE"), "."),
        # Installs WhisperX into its own environment. Run by the installer,
        # and again from File > Settings if it is ever needed a second time -
        # so it has to travel with the app, not just live in the installer.
        (os.path.join(SPECPATH, "setup_whisperx.ps1"), "."),
        # The window icon. The `icon=` on EXE below is the shell icon for the
        # .exe file itself and does nothing for the tkinter window.
        (os.path.join(SRC, "assets"), "assets"),
    ] + _bundled_plugins(),
    # These are reached only through the frozen re-launch or lazily inside
    # functions, so PyInstaller's import scan does not see them.
    hiddenimports=[
        "plugin_editor",
        # Imported only inside functions, so PyInstaller's import scan can't
        # see them - the problem report, the freeze trace (vst_host/
        # fx_dialog), the Sync dialog/render, and VAD-based cut detection.
        "diagnostics",
        "report_dialog",
        "sync_dialog",
        "sync_render",
        "silero_vad_onnx",
        "ollama_client",
        "sounddevice",
        "pedalboard",
        "numba",
        "llvmlite",
    ],
    hookspath=[],
    runtime_hooks=[],
    # WhisperX and friends stay external. Excluding them keeps an accidental
    # torch install in the build environment from adding gigabytes.
    excludes=[
        "torch", "torchaudio", "torchvision", "whisperx", "transformers",
        "matplotlib", "scipy", "pandas", "IPython", "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Wavefield",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                     # UPX compression is another antivirus trigger
    version=_version_resource(),
    console=False,                 # a GUI app: no console window behind it
    icon=os.path.join(SPECPATH, "autocut.ico")
         if os.path.exists(os.path.join(SPECPATH, "autocut.ico")) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Wavefield",
)
