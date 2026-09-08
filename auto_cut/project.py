"""
Saving and reopening a session.

Everything the app holds that isn't re-derivable from the media goes in a JSON
`.autocut` file: the recordings, every hand edit, mixer state, VST chains (with
each plugin's own `raw_state`, so a chain reopens exactly as it was tuned),
framing, scene overrides and the transcript.

Deliberately NOT stored: waveform peaks, decoded PCM and WhisperX output. Those
live in `.cache/`, are keyed by file content, and rebuild on demand - putting
them here would make projects enormous for no gain.
"""

import base64

import chain_io
import json
import os
import settings
import time

FORMAT_VERSION = 2
PROJECT_EXTENSION = ".autocut"


# Chain (de)serialisation lives in chain_io now - presets need exactly the same
# format, and one copy is the only way the two cannot drift apart. Re-exported
# under the old names because app_actions calls _chain_from_dict directly.
_chain_to_dict = chain_io.to_dict
_chain_from_dict = chain_io.from_dict


def build(app):
    """Snapshots the app into a plain dict."""
    return {
        "format_version": FORMAT_VERSION,
        "speaker_paths": list(app.speaker_paths),
        "aggressiveness": int(app.aggressiveness.get()),
        "language": app.language.get() if hasattr(app, "language") else None,
        "whisper_model": (app.whisper_model.get()
                          if hasattr(app, "whisper_model") else None),
        "auto_mute": bool(app.auto_mute_on.get()),
        "edits": [list(e) for e in app.edits],
        "mute_edits": [list(e) for e in app.mute_edits],
        "tracks": [
            {"gain": t.gain, "muted": bool(t.muted), "soloed": bool(t.soloed)}
            for t in app.player.tracks
        ],
        "chains": [_chain_to_dict(c) for c in app.track_chains],
        "intro_path": app.intro_path,
        "outro_path": app.outro_path,
        "export_stems": bool(app.export_stems.get()),
        # Speech is measured from the waveform, and re-measuring it means
        # decoding and denoising every track again - minutes of work for a
        # result that cannot have changed. The transcript is stored for the
        # same reason.
        "speech": [[list(iv) for iv in speech]
                   for speech in (getattr(app, "per_speaker_speech", None) or [])],
        "transcript": getattr(app, "transcript", None),
        "framing": getattr(app, "framing", None),
        "auto_cut": bool(app.auto_cut_on.get()) if hasattr(app, "auto_cut_on") else True,
        "words": [list(words) for words in (getattr(app, "per_speaker_words", None) or [])],
        # Vodcast. scene_edits matter most of these - they are hand work that
        # re-running analysis cannot recover.
        "v3_path": getattr(app, "v3_path", None),
        "scene_switching": (bool(app.scene_switching.get())
                            if hasattr(app, "scene_switching") else False),
        "min_shot_seconds": (float(app.min_shot_seconds.get())
                             if hasattr(app, "min_shot_seconds") else 2.0),
        "max_shot_seconds": (float(app.max_shot_seconds.get())
                             if hasattr(app, "max_shot_seconds") else 25.0),
        "scene_edits": [list(e) for e in getattr(app, "scene_edits", [])],
    }


def save(path, app):
    if not path.lower().endswith(PROJECT_EXTENSION):
        path += PROJECT_EXTENSION
    data = build(app)
    # Write via a temp file so an interrupted save can't destroy a good project.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "speaker_paths" not in data:
        raise ValueError("That doesn't look like an auto_cut project file.")
    return data


def missing_media(data):
    """Which referenced files are no longer where the project says they are."""
    missing = [p for p in data.get("speaker_paths", []) if not os.path.exists(p)]
    for key in ("intro_path", "outro_path"):
        p = data.get(key)
        if p and not os.path.exists(p):
            missing.append(p)
    return missing


def relink(data, replacements):
    """
    Applies {old_path: new_path} across the project. Used when a project is
    opened on a machine where the recordings have moved.
    """
    data["speaker_paths"] = [replacements.get(p, p) for p in data.get("speaker_paths", [])]
    for key in ("intro_path", "outro_path"):
        if data.get(key) in replacements:
            data[key] = replacements[data[key]]
    return data


# ------------------------------------------------------------------ autosave

AUTOSAVE_NAME = "recovery.autocut"
AUTOSAVE_SECONDS = 60


def autosave_path():
    """Beside the cache, not beside the user's project - it's scratch state."""
    directory = settings.cache_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, AUTOSAVE_NAME)


def autosave(app):
    """
    Snapshots the session so a crash doesn't cost the work.

    Written periodically rather than on exit: a native crash - a VST taking the
    process down, for instance - runs no Python cleanup at all, so there is no
    "on crash" hook to hang this off.
    """
    path = autosave_path()
    data = build(app)
    data["autosaved_at"] = time.time()
    data["autosaved_project"] = getattr(app, "project_path", None)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def pending_recovery(max_age_days=7):
    """
    A usable autosave from a previous run, or None.

    Ignores an empty one (nothing was loaded) and anything stale.
    """
    path = autosave_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not data.get("speaker_paths"):
        return None
    age = time.time() - float(data.get("autosaved_at", 0))
    if age > max_age_days * 86400:
        return None
    return data


def clear_autosave():
    try:
        os.remove(autosave_path())
    except OSError:
        pass
