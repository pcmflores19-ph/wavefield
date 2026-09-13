"""
Builds the file someone sends when Wavefield misbehaves.

The problems that actually reach us are environmental - a driver too old for
the bundled ffmpeg, a WhisperX whose torch has no CUDA, a plugin that kills the
interpreter outright - and every one of them is invisible in a description like
"it didn't work". This gathers the handful of facts that would otherwise take a
dozen messages to establish.

Two rules it follows:

  Never fail. A diagnostic report that raises while collecting diagnostics is
  worse than useless, so every probe is wrapped and missing information is
  recorded as a line saying so.

  Never include a full path from the user's disk. Recording names is enough to
  work out what went wrong, and this file is written to be emailed to a
  stranger - real paths carry the person's name, employer and folder layout.
  Only basenames go in, and the one place a full path is genuinely needed (the
  WhisperX location, which is the thing most likely to be wrong) is included
  deliberately because it cannot be diagnosed otherwise.
"""

import faulthandler
import os
import platform
import re
import subprocess
import sys
import threading
import time

import bundled
import settings
import version

NL = chr(10)


# ----------------------------------------------------------- the freeze trace
#
# A second log, separate from autocut_crash.log, and written for one specific
# failure: the app stops responding to clicks while audio carries on playing.
#
# app.log() cannot report that. It is a queue drained by _poll_log_queue on a
# root.after loop, so the moment the Tk event loop stops servicing callbacks -
# which is the thing being diagnosed - nothing reaches the log panel or the
# status bar at all. Everything here writes straight to a file and flushes, so
# it survives both a dead event loop and a native crash that runs no cleanup.
FREEZE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "autocut_freeze.log")

_trace_lock = threading.Lock()

# Trimmed to this at the start of every session, keeping the most recent part.
# autocut_crash.log reached 10,497 entries and became a red herring that cost
# an investigation real time; an unbounded diagnostic is a liability, not an
# asset.
FREEZE_LOG_MAX_BYTES = 1024 * 1024

# Set when the UI is shutting down, so the watchdog stops. Without this the
# heartbeat goes stale the moment mainloop() returns and every normal exit wrote
# a false "EVENT LOOP STALLED" plus a thread dump - a fake freeze at the end of
# every session, in the one file someone reads to find a real one.
_watchdog_stop = threading.Event()

# Last time the Tk thread proved it was alive, as a monotonic clock reading.
# Written only by the Tk thread, read only by the watchdog - a single float, so
# no lock is needed for that.
_last_heartbeat = [0.0]


def trace(message):
    """
    One timestamped line in the freeze log. Never raises: this is called from
    the audio path's neighbours and from a watchdog, and a failure to write a
    diagnostic must not become the failure being diagnosed.
    """
    try:
        stamp = time.strftime("%H:%M:%S")
        with _trace_lock:
            with open(FREEZE_LOG, "a", encoding="utf-8") as handle:
                handle.write(f"{stamp} {message}{NL}")
                handle.flush()
    except Exception:
        pass


def _rotate_if_large():
    """
    Keeps the tail and drops the rest. Best effort; never raises.

    Under the same lock as trace(), so a worker writing a line while this
    rewrites the file cannot lose it or land it half-written.
    """
    try:
        with _trace_lock:
            _rotate_locked()
    except Exception:
        pass


def _rotate_locked():
    try:
        if not os.path.exists(FREEZE_LOG):
            return
        size = os.path.getsize(FREEZE_LOG)
        if size <= FREEZE_LOG_MAX_BYTES:
            return
        keep = FREEZE_LOG_MAX_BYTES // 2
        with open(FREEZE_LOG, "rb") as handle:
            handle.seek(size - keep)
            tail = handle.read()
        with open(FREEZE_LOG, "wb") as handle:
            handle.write(b"(earlier entries trimmed)" + NL.encode("ascii"))
            handle.write(tail)
    except Exception:
        pass


def heartbeat():
    """Called from the Tk thread to say the event loop is still turning."""
    _last_heartbeat[0] = time.monotonic()


def stop_event_loop_watchdog():
    """
    Call when the UI is coming down, before the process exits.

    Not optional. `mainloop()` returning stops the heartbeat, so a watchdog left
    running decides the event loop has frozen and writes a stall plus a full
    thread dump on every clean exit. Reproduced: a 1.5-second session that quit
    normally still logged "EVENT LOOP STALLED - no Tk callback for 2.5s".
    """
    _watchdog_stop.set()


def start_event_loop_watchdog(root, stall_seconds=5.0, poll_seconds=1.0):
    """
    Notices when the Tk event loop stops turning, and records what every thread
    was doing at that moment.

    Two halves. A `root.after` loop stamps `_last_heartbeat`, and a plain daemon
    thread - deliberately NOT a Tk callback, because a Tk callback is exactly
    what stops running - checks how old that stamp is. Past `stall_seconds` it
    writes the stall and a full all-threads traceback to FREEZE_LOG, then waits
    for the loop to come back and records how long it was gone.

    This is what the 2026-09-12 investigation lacked. `py-spy dump` on the
    frozen process showed the main thread idle in `mainloop` with no app frames
    below it, which says the loop was not stuck inside app code but not what
    stopped it, and it could only be taken by hand after the fact. Here the
    stacks are captured at the instant the stall starts, including the worker
    threads that py-spy's two dumps could not be aimed at.

    `stall_seconds` is deliberately generous. Windows runs its own modal loop
    while a window is being dragged or resized, which stops Tk timer callbacks
    for as long as the mouse is held - a short threshold reports that as a
    freeze. The bug this exists for does not end, so nothing is lost by waiting.
    """
    heartbeat()

    def tick():
        heartbeat()
        try:
            root.after(int(poll_seconds * 1000), tick)
        except Exception:
            pass            # shutting down

    def watch():
        stalled_since = None
        while not _watchdog_stop.wait(poll_seconds):
            gap = time.monotonic() - _last_heartbeat[0]
            if gap > stall_seconds and stalled_since is None:
                stalled_since = time.monotonic()
                trace(f"EVENT LOOP STALLED - no Tk callback for {gap:.1f}s. "
                      f"All thread stacks follow.")
                try:
                    with _trace_lock:
                        with open(FREEZE_LOG, "a", encoding="utf-8") as handle:
                            faulthandler.dump_traceback(file=handle,
                                                        all_threads=True)
                            handle.write(NL)
                            handle.flush()
                except Exception:
                    pass
            elif gap <= stall_seconds and stalled_since is not None:
                held = time.monotonic() - stalled_since
                stalled_since = None
                trace(f"event loop recovered after {held:.1f}s")

    # Cleared here, not only at import: a stop flag left set from an earlier
    # run would silently disarm the watchdog on the next start.
    _watchdog_stop.clear()
    _rotate_if_large()
    try:
        root.after(int(poll_seconds * 1000), tick)
        threading.Thread(target=watch, daemon=True,
                         name="event-loop-watchdog").start()
    except Exception:
        pass


def _freeze_lines(limit=200):
    """The tail of the freeze log, for the report."""
    if not os.path.exists(FREEZE_LOG):
        return ["(no freeze log - no stall has been recorded)"]
    try:
        with open(FREEZE_LOG, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except Exception as exc:
        return [f"could not read the freeze log ({exc})"]
    if not lines:
        return ["(freeze log is empty)"]
    return [_strip_paths(line) for line in lines[-limit:]]


def _run(command, timeout=15):
    """First line of a command's output, or a note explaining its absence."""
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout,
                                creationflags=getattr(subprocess,
                                                      "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        return "not found"
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"
    except Exception as exc:
        return f"could not run ({exc})"
    text = (result.stdout or result.stderr or "").strip()
    return text.splitlines()[0] if text else f"no output (exit {result.returncode})"


def _section(title):
    return NL + title + NL + "-" * len(title)


def _app_lines():
    lines = [f"{version.APP_NAME} {version.__version__}",
             f"frozen build: {bundled.frozen()}"]
    return lines


def _system_lines():
    return [
        f"os        : {platform.platform()}",
        f"machine   : {platform.machine()}",
        f"python    : {sys.version.split()[0]}",
    ]


def _gpu_lines():
    out = _run(["nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"])
    return [f"nvidia-smi: {out}"]


def _tool_lines():
    lines = []
    for name in ("ffmpeg", "ffprobe"):
        path = bundled.tool(name)
        # Whether it is the bundled copy or one from PATH is exactly the
        # distinction that explains "works on my machine".
        which = "bundled" if os.path.isabs(path) else "from PATH"
        lines.append(f"{name:9}: {_run([path, '-version'])}  ({which})")
    return lines


def _whisperx_lines():
    lines = []
    configured = settings.get("whisperx_path") or "(not set - searched for)"
    lines.append(f"configured: {configured}")
    try:
        import whisperx_runner
        path, has_cuda = whisperx_runner.resolve()
        lines.append(f"found     : {path or 'nothing found'}")
        lines.append(f"can use GPU: {has_cuda}")
        lines.append(f"device    : {whisperx_runner.device()}")
    except Exception as exc:
        lines.append(f"lookup failed: {exc}")
    return lines


def _package_lines():
    lines = []
    for name in ("numpy", "sounddevice", "pedalboard"):
        try:
            module = __import__(name)
            lines.append(f"{name:12}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            lines.append(f"{name:12}: not importable ({exc})")
    return lines


def _project_lines(app):
    """
    What the session was doing - names only, never full paths.
    """
    lines = []
    try:
        paths = getattr(app, "speaker_paths", []) or []
        lines.append(f"recordings loaded: {len(paths)}")
        for index, path in enumerate(paths):
            lines.append(f"  {index + 1}. {os.path.basename(path)}")
        media = getattr(app, "speaker_media", None) or []
        for index, info in enumerate(media):
            lines.append(f"  {index + 1}. {info.duration_seconds:.1f}s, "
                         f"video={info.has_video}")
        lines.append(f"timeline duration: {getattr(app, 'timeline_duration', 0):.1f}s")
        lines.append(f"transcript segments: "
                     f"{len((getattr(app, 'transcript', None) or {}).get('segments', []))}")
        lines.append(f"camera switching: {_var(app, 'scene_switching')}")
        lines.append(f"auto-cut: {_var(app, 'auto_cut_on')}  "
                     f"auto-mute: {_var(app, 'auto_mute_on')}")
    except Exception as exc:
        lines.append(f"could not read session state: {exc}")
    return lines


def _var(app, name):
    try:
        return getattr(app, name).get()
    except Exception:
        return "unknown"


def _log_lines(app, limit=400):
    try:
        text = app.log_text.get("1.0", "end").strip()
    except Exception as exc:
        return [f"could not read the log ({exc})"]
    if not text:
        return ["(empty)"]
    lines = text.splitlines()
    if len(lines) > limit:
        lines = [f"... {len(lines) - limit} earlier lines omitted ..."] + lines[-limit:]
    return lines


# Matches an absolute Windows, UNC or POSIX path. faulthandler dumps are full
# of them, and this module's contract (see the header) is that a report
# carries basenames only - it is written to be emailed to a stranger.
#
# The POSIX branch insists on at least one directory component, so a bare
# slash inside ordinary prose is left alone. Without that it rewrote
# "NVIDIA/Intel" to "NVIDIAIntel" and "n/a" to "na" - a diagnostic quietly
# corrupting its own evidence, which is the exact failure this file exists
# to avoid.
_ABSOLUTE_PATH = re.compile(r"""(?:[A-Za-z]:[\\/](?:[^\s"']*[\\/])?|\\\\(?:[^\s"']*[\\/])?|/(?:[^\s"'/]+/)+)([^\\/\s"']+)""")


def _strip_paths(line):
    """Replaces every absolute path in a line with just its basename."""
    return _ABSOLUTE_PATH.sub(lambda m: m.group(1), line)


def _crash_lines(limit=120):
    """
    The tail of the native crash log, which is the whole reason it exists.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "autocut_crash.log")
    if not os.path.exists(path):
        return ["(no crash log - the app has not died unexpectedly)"]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except Exception as exc:
        return [f"could not read the crash log ({exc})"]
    if not lines:
        return ["(crash log is empty)"]
    return [_strip_paths(line) for line in lines[-limit:]]


def summary(app=None, description=None):
    """The whole report as one string."""
    parts = []
    parts.append(f"{version.APP_NAME} problem report")
    parts.append(time.strftime("%Y-%m-%d %H:%M:%S"))

    # First, prominently, and before any technical detail: what the person
    # actually typed. A GPU model and a package list don't say what broke -
    # this is the part of the report that does.
    parts.append(_section("What happened"))
    parts.append(description if description else "(no description given)")

    for title, lines in (
            ("Application", _app_lines()),
            ("System", _system_lines()),
            ("Graphics", _gpu_lines()),
            ("Bundled tools", _tool_lines()),
            ("Speech recognition", _whisperx_lines()),
            ("Python packages", _package_lines()),
    ):
        parts.append(_section(title))
        parts.extend(lines)

    if app is not None:
        parts.append(_section("This session"))
        parts.extend(_project_lines(app))

    parts.append(_section("Log"))
    parts.extend(_log_lines(app) if app is not None else ["(not available)"])

    parts.append(_section("Crash log"))
    parts.extend(_crash_lines())

    # After the crash log on purpose: a stall that never became a crash leaves
    # nothing in the crash log at all, and this is the only record of it.
    parts.append(_section("Event loop stalls"))
    parts.extend(_freeze_lines())

    return NL.join(parts) + NL


def write_report(app=None, directory=None, description=None):
    """
    Writes the report and returns its path.

    Lands in the settings folder rather than beside the program: on a normal
    install the program directory is not writable, and that is exactly the
    moment someone is trying to report a problem.
    """
    directory = directory or settings.config_dir()
    os.makedirs(directory, exist_ok=True)
    name = time.strftime("wavefield-report-%Y%m%d-%H%M%S.txt")
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(summary(app, description))
    return path


def reveal(path):
    """Opens the folder containing `path`, selecting it where possible."""
    folder = os.path.dirname(path)
    try:
        if os.name == "nt":
            # explorer returns a non-zero exit code even when it works, so its
            # result is deliberately not checked.
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", folder])
        return True
    except Exception:
        return False
