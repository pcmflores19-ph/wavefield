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
import time

APP_DIR_NAME = "AutoCut"
FILE_NAME = "settings.json"

# The one thing in cache_dir() that is not a rebuildable cache - project.py's
# crash-recovery autosave. Named here rather than imported (project.py
# already imports settings; importing it back would be circular) so pruning
# and "Clear cache" can never touch it.
_PROTECTED_CACHE_FILES = {"recovery.autocut"}

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


def _cache_entries():
    directory = cache_dir()
    if not os.path.isdir(directory):
        return []
    entries = []
    for name in os.listdir(directory):
        if name in _PROTECTED_CACHE_FILES:
            continue
        path = os.path.join(directory, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if os.path.isfile(path):
            entries.append([path, st.st_size, st.st_atime, st.st_mtime])
    return entries


def clear_cache():
    """
    Deletes every regenerable cache file (decoded audio, cleaned audio,
    transcripts) - never the crash-recovery autosave. Best-effort per file: a
    file another part of the app is still using (memory-mapped for playback,
    say) is skipped rather than raised as an error, the same way every other
    cache access here is.

    Returns (files_removed, bytes_freed, files_skipped) so a caller - the
    Settings dialog's "Clear cache" button - can report something concrete.
    """
    removed = freed = skipped = 0
    for path, size, _atime, _mtime in _cache_entries():
        try:
            os.remove(path)
            removed += 1
            freed += size
        except OSError:
            skipped += 1
    return removed, freed, skipped


def prune_cache(max_total_bytes=8_000_000_000, max_age_days=14):
    """
    Keeps the analysis cache from growing forever, without ever deleting the
    crash-recovery autosave or a file something else has locked open.

    Two passes: first, anything older than max_age_days goes regardless of
    size (a cache entry nobody has touched in two weeks is not earning its
    disk space); then, if still over max_total_bytes, the least-recently-used
    survivors go next until back under budget. A locked file (memory-mapped
    for playback, most likely) is left in place either way - the OS itself
    refuses that delete, this just doesn't treat it as an error.

    Default raised from 2GB (2026-09-11): a single 3-speaker, ~1 hour session
    alone produces ~2.3GB of .mono.pcm + .clean16.pcm (measured), so the old
    default pruned a session's OWN cache on the very next launch, forcing a
    full needless re-decode. 8GB comfortably covers several sessions like
    that.

    Meant to run once, early, in a background thread - see
    app._prune_cache_worker - not on any hot path.
    """
    entries = _cache_entries()
    if not entries:
        return

    now = time.time()
    survivors = []
    for entry in entries:
        path, size, atime, mtime = entry
        if (now - mtime) / 86400.0 > max_age_days:
            try:
                os.remove(path)
                continue
            except OSError:
                pass                     # in use - leave it, count it below
        survivors.append(entry)

    total = sum(e[1] for e in survivors)
    if total <= max_total_bytes:
        return
    survivors.sort(key=lambda e: e[2])   # oldest-accessed first
    for path, size, _atime, _mtime in survivors:
        if total <= max_total_bytes:
            break
        try:
            os.remove(path)
            total -= size
        except OSError:
            continue                     # in use - skip, try the next oldest
