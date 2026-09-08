"""
Settings that belong to this computer rather than to an episode.

Projects already save everything about a recording - files, edits, levels,
effect chains. What they cannot save is where WhisperX lives, because that is
a property of the machine: copy a project to another computer and the path
would be wrong.

Small and best-effort by design. A settings file that cannot be read must never
stop the app starting; it just falls back to defaults.
"""

import json
import os

APP_DIR_NAME = "AutoCut"
FILE_NAME = "settings.json"

DEFAULTS = {
    # Blank means "find it yourself". A path here is a deliberate override for
    # someone who installed WhisperX somewhere unusual - which is common,
    # because installing it into its own virtualenv is the sensible way to do
    # it and that is never on the PATH.
    "whisperx_path": "",

    # Whether to ask GitHub for a newer release when the app opens. On by
    # default, but a real setting rather than a hidden behaviour: this is the
    # only network request Wavefield ever makes, and someone who wants a
    # machine that talks to nothing should be able to have one. Help > Check
    # for updates still works either way.
    "check_updates_on_start": True,
}


def config_dir():
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_DIR_NAME)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, APP_DIR_NAME.lower())


def cache_dir():
    """
    Where decoded audio and transcripts are cached - regenerable, and can run
    to hundreds of MB an episode, so LOCALAPPDATA (never synced) rather than
    the small roaming settings.json above.

    Must never be beside the program itself: an installed copy sits under
    Program Files, which a standard user cannot write to, and PyInstaller's
    __file__ points inside it.
    """
    if os.name == "nt":
        base = (os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
                 or os.path.expanduser("~"))
        return os.path.join(base, APP_DIR_NAME, "cache")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, APP_DIR_NAME.lower())


def config_path():
    return os.path.join(config_dir(), FILE_NAME)


_cache = None


def load(force=False):
    """Every setting, defaults filled in for anything missing."""
    global _cache
    if _cache is not None and not force:
        return _cache

    values = dict(DEFAULTS)
    try:
        # utf-8-sig, not utf-8: the WhisperX setup script writes this file
        # from PowerShell, and Windows PowerShell 5.1 puts a BOM on everything
        # it writes. Plain utf-8 chokes on that, and because this whole read is
        # best-effort the failure is silent - the app would quietly forget
        # where WhisperX was installed. utf-8-sig reads both.
        with open(config_path(), "r", encoding="utf-8-sig") as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            # Only keys we know about: an old or hand-edited file should not be
            # able to inject anything unexpected.
            for key in DEFAULTS:
                if key in stored:
                    values[key] = stored[key]
    except Exception:
        pass                     # missing or unreadable is the normal case
    _cache = values
    return _cache


def get(key):
    return load().get(key, DEFAULTS.get(key))


def set_value(key, value):
    """Writes one setting through to disk. Returns True if it was saved."""
    values = load()
    values[key] = value
    try:
        os.makedirs(config_dir(), exist_ok=True)
        with open(config_path(), "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2)
        return True
    except Exception:
        return False             # read-only profile, locked file, full disk
