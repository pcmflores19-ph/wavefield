"""
VST3 hosting: plugin discovery and per-track effect chains, via pedalboard.

Each speaker gets their own chain, since one mic may only need a denoiser while
another needs denoise + de-ess + leveling.

Discovery gotcha: most VST3s ship as *bundles* - a `Name.vst3` directory with
the real binary at `Contents/x86_64-win/Name.vst3`. Handing pedalboard the
directory fails with "unsupported plugin format", so we resolve to the inner
binary. A few plugins are plain .vst3 files and are used as-is.
"""

import base64
import contextlib
import os
import platform
import queue
import struct
import subprocess
import tempfile
import threading
import time

import numpy

import bundled

# How much audio a plugin gets at a time when it could not manage the whole
# track at once. Long enough that the joins are rare, short enough that any
# plugin's declared maximum block size is comfortably clear.
CHUNK_SECONDS = 30.0

def _default_search_dirs():
    """
    The standard VST3 locations for this operating system.

    The VST3 spec puts plugins in fixed places on each platform, so there is
    nothing to configure in the normal case - and `extra_dirs` covers the rest.
    """
    system = platform.system()
    if system == "Windows":
        return [
            r"C:\Program Files\Common Files\VST3",
            r"C:\Program Files\VST3",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Common\VST3"),
            os.path.expandvars(r"%COMMONPROGRAMFILES%\VST3"),
        ]
    if system == "Darwin":
        return [
            "/Library/Audio/Plug-Ins/VST3",
            os.path.expanduser("~/Library/Audio/Plug-Ins/VST3"),
        ]
    return [
        os.path.expanduser("~/.vst3"),
        "/usr/lib/vst3",
        "/usr/local/lib/vst3",
    ]


def _search_dirs():
    """
    System VST3 folders, then any plugins shipped with the app.

    Order matters. discover_plugins keeps the FIRST match for a given name, so
    putting the bundle last means a user's own install of rnnoise (or anything
    else we ship) wins over our copy - which is what they would expect.
    """
    dirs = _default_search_dirs()
    shipped = bundled.vst3_dir()
    if shipped:
        dirs.append(shipped)
    return dirs


VST3_SEARCH_DIRS = _search_dirs()

# Inside a bundle the binary sits under Contents/<arch>/. The names differ per
# platform, and listing all of them costs nothing - a directory that is not
# there is simply skipped.
_ARCH_DIRS = [
    "x86_64-win", "x86-win", "arm64-win",        # Windows
    "MacOS",                                      # macOS
    "x86_64-linux", "aarch64-linux",              # Linux
    "Contents",
]

class GateUnavailable(RuntimeError):
    """
    The plugin gate could not be taken in time, or the caller gave up first.

    Raised instead of waiting indefinitely. Every caller already degrades
    gracefully on an exception - a processing pass hands back the unprocessed
    audio, a load reports that the copy could not be made - so this surfaces
    as one missed block or one failed Apply rather than as the frozen app an
    unbounded wait produced.
    """


# How long a load may wait for in-flight processing passes to drain. A normal
# drain is one audio block (~23ms) or one waveform chunk; anything approaching
# this means something long-running overlapped, which the callers are supposed
# to prevent. Deliberately shorter than _MAIN_THREAD_LOAD_TIMEOUT_SECONDS
# below, so the main thread frees itself before the worker waiting on it gives
# up - the other way round leaves the main thread asleep with nobody left to
# wake it.
_LOAD_GATE_TIMEOUT_SECONDS = 10.0

# How long a processing pass may wait for a pending load. A load is short, so
# this only has to outlast one; a pass that hits it is one block of audio
# passed through unprocessed, never a stall.
_PROCESS_GATE_TIMEOUT_SECONDS = 5.0

# How often a waiter re-checks its abort predicate while parked on the
# condition variable.
_GATE_POLL_SECONDS = 0.05


class _ProcessLoadGate:
    """
    Lets any number of processing passes run at once, but never while a plugin
    is being loaded.

    Loading a VST while another thread is inside one is a native crash, not an
    exception: pedalboard aborts the process with "PyEval_RestoreThread: the
    GIL is released" and no try/except can catch it. So loads must be exclusive.

    Processing passes, though, only ever run on plugin instances owned by their
    own chain - the audio callback on the live chain, a waveform redraw or an
    export on a private copy from TrackChain.snapshot(). Nothing is shared, so
    they can safely overlap, and they must: a plain mutex here made the audio
    callback wait out an entire offline pass, a ten-second stall for what should
    be a 23-millisecond block.

    Two properties beyond that exclusion, both added after the freeze diagnosed
    on 2026-09-12, where "Apply to whole track" left the app permanently
    unclickable while audio carried on playing:

    * **A pending load has priority.** The original `processing()` only waited
      while a load was already under way, and `loading()` only became "under
      way" once it had won. So a load waiting for `_processing` to reach zero
      could be starved indefinitely by the audio callback, which re-enters this
      gate every 23ms for as long as playback runs. `_load_waiting` marks the
      intent to load, and new processing passes queue behind it, so the count
      genuinely drains.

    * **Nobody waits forever.** The load hop runs on the Tk main thread (see
      `_load_plugin_on_main_thread`), so an unbounded wait here is an unbounded
      freeze of the whole UI - with no event loop left to cancel it, close a
      dialog, or stop playback. Both waits are bounded and raise
      `GateUnavailable` instead.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._processing = 0
        self._loading = False
        # Loads that have declared themselves but not yet won the gate. A
        # count, not a flag: two threads can be waiting to load at once (a
        # waveform redraw's snapshot and an Apply's, say) and the first to
        # finish must not clear the other's priority.
        self._load_waiting = 0

    @contextlib.contextmanager
    def processing(self, timeout=_PROCESS_GATE_TIMEOUT_SECONDS):
        """
        Admits a processing pass, waiting out any load that is running OR
        waiting. Raises GateUnavailable rather than blocking past `timeout`.
        """
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._loading or self._load_waiting:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GateUnavailable(
                        "timed out waiting for a plugin load to finish")
                self._cv.wait(remaining)
            self._processing += 1
        try:
            yield
        finally:
            with self._cv:
                self._processing -= 1
                self._cv.notify_all()

    @contextlib.contextmanager
    def loading(self, timeout=_LOAD_GATE_TIMEOUT_SECONDS, abort=None):
        """
        Takes the gate exclusively for a plugin load.

        `abort` is an optional predicate re-checked while waiting; the caller
        that queued this load uses it to release the waiter once it has given
        up (see `_load_plugin_on_main_thread`). Raises GateUnavailable on
        either abort or timeout - it never blocks indefinitely, because this
        can run on the Tk main thread.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            # Declared before the wait, so processing passes queue behind it
            # from this moment rather than continuing to slip in ahead.
            self._load_waiting += 1
            self._cv.notify_all()
            try:
                while self._loading or self._processing:
                    if abort is not None and abort():
                        raise GateUnavailable(
                            "the caller gave up before the gate was free")
                    if deadline is None:
                        self._cv.wait(_GATE_POLL_SECONDS)
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise GateUnavailable(
                            "timed out waiting for processing to drain")
                    # Capped so `abort` is re-checked promptly even when the
                    # remaining budget is long.
                    self._cv.wait(min(remaining, _GATE_POLL_SECONDS))
                self._loading = True
            finally:
                # Won, aborted or timed out, this waiter is no longer pending -
                # leaving it counted would block processing forever.
                self._load_waiting -= 1
                self._cv.notify_all()
        try:
            yield
        finally:
            with self._cv:
                self._loading = False
                self._cv.notify_all()


# A thread that is already inside processing() must never try to load: that
# would wait on itself. Every caller here loads first, then processes.
_GATE = _ProcessLoadGate()


def _trace(message):
    """
    Into the freeze log, not the UI log.

    Everything in this module runs while the UI may be unresponsive, which is
    when app.log() - a queue drained by a root.after loop - reaches nobody.
    Imported lazily so importing vst_host stays cheap and cycle-free.
    """
    try:
        import diagnostics
        diagnostics.trace(message)
    except Exception:
        pass


# JUCE only lets a VST3 be instantiated on whichever thread first created its
# MessageManager - in this app, that is the main/Tk thread, since TrackChain.
# add() (the UI's "add effect" path) always runs there. snapshot() below loads
# plugins from a background worker thread instead, which pedalboard rejects
# outright ("must be reloaded on the main thread"). set_main_thread_runner
# lets app.py hand us a way to hop back onto the main thread for just the load.
_main_thread_runner = None


def set_main_thread_runner(runner):
    """`runner(fn)` must arrange for `fn()` to run on the app's main thread
    soon (e.g. `lambda fn: root.after(0, fn)`)."""
    global _main_thread_runner
    _main_thread_runner = runner


# The worker's give-up deadline. Longer than _LOAD_GATE_TIMEOUT_SECONDS on
# purpose: the main thread's own wait should expire first and free the event
# loop, leaving this as the backstop for a runner that never fires at all
# (the multiprocessing interaction described in _load_plugin_on_main_thread).
_MAIN_THREAD_LOAD_TIMEOUT_SECONDS = 30.0

# How long the hop may take before it is worth saying so in the log. Below this
# it is just a normal plugin load.
_HOP_REPORT_AFTER_SECONDS = 1.0


def _load_plugin_on_main_thread(path, log=None):
    """
    Loads a VST3 the way `add()` does, but safely from any thread.

    Reproduced directly (2026-09-11): repeatedly combining this hop with
    repeated multiprocessing.Process spawns elsewhere in the app (the
    isolated Silero VAD call, once per speaker) hangs `root.after(0, run)`
    from ever firing, by the 3rd such round - confirmed by elimination,
    removing only the hop made an otherwise-identical repro survive every
    time. The exact OS-level mechanism wasn't pinned down further than that,
    so `done.wait()` now has a timeout: a hang here becomes a normal,
    catchable exception instead of freezing the whole app forever. Callers
    that can (analysis) should also prefer loading in a dedicated child
    process instead of relying on this hop at all - see
    voice_activity._denoise_isolated - since a fresh process never touches
    the main thread's plugin-hosting state and isn't vulnerable to this at
    all, per the same reproduction.
    """
    import pedalboard
    if _main_thread_runner is None or threading.current_thread() is threading.main_thread():
        # Already on the thread that must do the load (TrackChain.add's path).
        # Still bounded: this IS the Tk thread, so an unbounded wait here is
        # the same frozen app as the hop below produced.
        with _GATE.loading():
            return pedalboard.load_plugin(path)

    result = {}
    done = threading.Event()
    cancelled = threading.Event()

    def run():
        # The gate is taken HERE, on the thread that actually loads, never by
        # the thread waiting below. Held the other way round it deadlocks
        # outright: the waiter holds the gate, the main thread blocks trying to
        # take it for its own load (TrackChain.add), and because the main
        # thread is blocked it never runs this closure - so the wait below
        # always ran its full timeout. Confirmed in autocut_crash.log.
        #
        # This closure runs on the Tk main thread, so every wait inside it is
        # bounded and `cancelled` is re-checked THROUGHOUT the wait, not only
        # before it. Checking only before it is what froze the app on
        # 2026-09-12: the waiter below timed out and set `cancelled`, but this
        # thread was already parked inside the gate and never looked again, so
        # the event loop stayed dead while playback carried on holding the
        # gate open.
        try:
            if cancelled.is_set():
                return
            with _GATE.loading(timeout=_LOAD_GATE_TIMEOUT_SECONDS,
                               abort=cancelled.is_set):
                if cancelled.is_set():
                    return
                plugin = pedalboard.load_plugin(path)
            if cancelled.is_set():
                # Nobody is waiting for this any more. Dropping the only
                # reference is the disposal - keeping it would leave a live
                # plugin instance nobody owns.
                del plugin
                return
            result["plugin"] = plugin
        except Exception as exc:
            result["error"] = exc
        finally:
            done.set()

    _main_thread_runner(run)
    # Waited in slices rather than one call, purely so a slow hop says so.
    # A silent 30-second stall here is indistinguishable from the app having
    # frozen, which is how the 2026-09-12 investigation lost time: the hop is
    # the known-hanging step (see the docstring) and it left no trace in the
    # log either way. The first slice covers the normal case without noise.
    name = os.path.basename(os.path.normpath(path)) or path
    waited = 0.0
    complained = False
    while waited < _MAIN_THREAD_LOAD_TIMEOUT_SECONDS:
        slice_seconds = min(_HOP_REPORT_AFTER_SECONDS,
                            _MAIN_THREAD_LOAD_TIMEOUT_SECONDS - waited)
        if done.wait(slice_seconds):
            break
        waited += slice_seconds
        if not complained:
            complained = True
            # Deliberately not "the UI is frozen": while this wait runs the
            # main thread is sitting in mainloop and the window responds
            # normally. It blocks only briefly later, when the queued closure
            # actually loads. Saying otherwise would put a false claim in the
            # one place someone looks to find a true one.
            message = (f"{name}: the main thread has not picked up the load "
                       f"yet ({waited:.0f}s). Effects are not applied until "
                       f"it does.")
            _trace(f"load hop: {message}")
            if log:
                log(message)
    if not done.is_set():
        message = (f"{name}: the main thread never loaded it "
                   f"({_MAIN_THREAD_LOAD_TIMEOUT_SECONDS:.0f}s). Giving up - "
                   f"this is the load hop hanging, not the render.")
        _trace(f"load hop: {message}")
        if log:
            log(message)
        # The closure is still queued and nothing else can cancel it. Left to
        # run it would load a plugin on the main thread long after this caller
        # gave up, outside any gate, while this thread is back inside
        # process() - two threads in pedalboard at once, which is the native
        # access violation this whole gate exists to prevent.
        cancelled.set()
        raise RuntimeError(
            f"Timed out waiting for the main thread to load {path} "
            f"(no response after {_MAIN_THREAD_LOAD_TIMEOUT_SECONDS:.0f}s)")
    if "error" in result:
        raise result["error"]
    return result["plugin"]


def _resolve_binary(entry_path):
    """Returns the loadable .vst3 path for a bundle directory, or the file itself."""
    if os.path.isfile(entry_path):
        return entry_path
    if not os.path.isdir(entry_path):
        return None
    contents = os.path.join(entry_path, "Contents")
    if os.path.isdir(contents):
        for arch in _ARCH_DIRS:
            arch_dir = os.path.join(contents, arch)
            if not os.path.isdir(arch_dir):
                continue
            for name in os.listdir(arch_dir):
                if name.lower().endswith(".vst3"):
                    return os.path.join(arch_dir, name)
    return None


def discover_plugins(extra_dirs=None):
    """
    Returns a sorted [(display_name, path)] of loadable VST3s found on this
    machine. Only paths are resolved here - plugins aren't loaded until used,
    since loading each one is slow.
    """
    found = {}
    for directory in list(VST3_SEARCH_DIRS) + list(extra_dirs or []):
        if not directory or not os.path.isdir(directory):
            continue
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            if not entry.lower().endswith(".vst3"):
                continue
            binary = _resolve_binary(os.path.join(directory, entry))
            if binary:
                found.setdefault(os.path.splitext(entry)[0], binary)
    return sorted(found.items(), key=lambda kv: kv[0].lower())


class PluginSlot:
    is_native = False

    def __init__(self, name, path, plugin):
        self.name = name
        self.path = path
        self.plugin = plugin
        self.bypassed = False
        # Held while audio is being processed, so state coming back from an
        # open editor is never applied mid-block.
        self.lock = threading.Lock()
        self.editor_process = None   # the open plugin GUI, if any
        # Non-None only while an editor is open: Player publishes this track's
        # live pre-chain audio here so the editor subprocess's OWN plugin
        # instance can process it too, purely so plugin GUIs with a live
        # meter/gain-reduction display (e.g. a de-esser) have real signal to
        # draw - the editor's instance otherwise never sees audio at all. See
        # open_editor_subprocess().
        self.editor_audio_queue = None
        # Set whenever params/bypass change after this slot was last baked
        # into a Track's processed_samples (see player.py) - tells the
        # incremental "Apply to whole track" cache it can no longer reuse
        # this slot's prior render. Cleared by app.py after a successful bake.
        self.dirty = False
        self.revision = 0

    def mark_dirty(self):
        self.dirty = True
        self.revision += 1

    def apply_state(self, raw_state):
        with self.lock:
            self.plugin.raw_state = raw_state
        self.mark_dirty()


class NativeSlot:
    """
    One of the built-in effects (see effects.py), in the same chain as any VST3.

    Deliberately the same shape as PluginSlot - name, bypassed, lock - so
    TrackChain does not have to care which kind it is holding. What differs is
    that this one has no external plugin to load, so it costs nothing to copy
    and cannot crash the process.
    """

    is_native = True

    def __init__(self, key, params=None):
        import effects
        self.key = key
        self.name = effects.BY_KEY[key][0]
        self.path = None
        self.params = dict(effects.defaults(key))
        if params:
            self.params.update(params)
        self.bypassed = False
        self.lock = threading.Lock()
        self.editor_process = None
        self.dirty = False
        self.revision = 0
        # This slot's own envelope/gate/filter memory, carried across
        # successive process() calls during continuous live playback (see
        # player.py's Player._reset_pending) - the whole point of this
        # attribute is that this NativeSlot object persists for the life of
        # the chain, unlike the audio buffer passed into process() each
        # block. Cleared only on reset=True (a real discontinuity - a seek,
        # a cut being skipped, or an offline render's single fresh pass).
        self._live_state = {}

    def mark_dirty(self):
        self.dirty = True
        self.revision += 1

    def process(self, audio, sample_rate, reset=False):
        import effects
        if reset:
            self._live_state = {}
        return effects.apply(self.key, audio, sample_rate, self.params,
                             state=self._live_state)

    def copy(self):
        return NativeSlot(self.key, self.params)


def _focus_editor_window(pid):
    """Raises an already-open plugin window rather than opening a second one."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        window_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid and user32.IsWindowVisible(hwnd):
            user32.ShowWindow(hwnd, 9)          # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return False
        return True

    try:
        user32.EnumWindows(enum_proc(callback), 0)
    except Exception:
        pass


def _forward_editor_audio(slot, proc):
    """
    Drains `slot.editor_audio_queue` (published from Player's realtime
    callback, see player.py's _mix_into) and streams each block to the editor
    subprocess's stdin as a length-prefixed float32 frame, so its own plugin
    instance can process() real audio and its meters have something to draw.

    Runs on its own thread - open_editor_subprocess's worker thread is fully
    occupied iterating proc.stdout for STATE: lines. Never touches slot.plugin
    directly, only the queue and the subprocess's stdin, so it needs none of
    slot.lock's protection.
    """
    while True:
        try:
            block = slot.editor_audio_queue.get(timeout=0.5)
        except queue.Empty:
            if proc.poll() is not None:
                return
            continue
        if block is None:
            return
        try:
            frame = struct.pack("<I", block.size) + block.astype(numpy.float32).tobytes()
            proc.stdin.buffer.write(frame)
            proc.stdin.buffer.flush()
        except Exception:
            # Broken pipe once the child exits, or any other write failure -
            # this thread's only job is a cosmetic meter, never let it raise
            # into anything that matters.
            return


def open_editor_subprocess(slot, on_done=None, on_error=None):
    """
    Opens `slot`'s plugin editor in a separate process, seeded with the plugin's
    current state, and applies whatever the user changed back to the live
    instance when they close the window.

    pedalboard can only show a plugin UI from the main thread, and blocks until
    it closes - doing that in-process would freeze the whole app, so the editor
    is hosted out-of-process instead. Returns immediately; callbacks fire on a
    worker thread.
    """
    # Already open? Bring that window forward instead of spawning another.
    existing = getattr(slot, "editor_process", None)
    if existing is not None and existing.poll() is None:
        _focus_editor_window(existing.pid)
        if on_error:
            on_error("editor already open - brought it to the front")
        return

    def run():
        state_file = None
        applied_any = False
        try:
            try:
                with slot.lock:
                    current = slot.plugin.raw_state
            except Exception:
                current = b""

            if current:
                fd, state_file = tempfile.mkstemp(prefix="autocut_vststate_", suffix=".bin")
                with os.fdopen(fd, "wb") as f:
                    f.write(current)

            cmd = bundled.editor_command(slot.path, state_file)

            # Sample rate for the audio this editor's OWN plugin instance will
            # be fed (see _forward_editor_audio below) - via env, not argv, so
            # it can't disturb editor_command()'s existing optional
            # state_file positional argument or app.py's frozen-relaunch
            # sys.argv handling.
            from player import SAMPLE_RATE
            child_env = dict(os.environ)
            child_env["AUTOCUT_EDITOR_SAMPLE_RATE"] = str(SAMPLE_RATE)

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, bufsize=1,
                                    env=child_env)
            slot.editor_process = proc
            slot.editor_audio_queue = queue.Queue(maxsize=2)
            threading.Thread(target=_forward_editor_audio, args=(slot, proc),
                             daemon=True).start()

            # The editor streams state as the user turns knobs, so apply each
            # update straight away - that's what makes the change audible live
            # instead of only after the window closes.
            for line in proc.stdout:
                if not line.startswith("STATE:"):
                    continue
                try:
                    slot.apply_state(base64.b64decode(line[len("STATE:"):].strip()))
                    applied_any = True
                except Exception as exc:
                    if on_error:
                        on_error(f"could not apply edited state: {exc}")

            proc.wait()
            stderr = (proc.stderr.read() or "").strip()

            if proc.returncode != 0 and not applied_any:
                if on_error:
                    on_error(stderr or f"exit code {proc.returncode}")
                return
            if on_done:
                on_done()
        except Exception as exc:
            if on_error:
                on_error(str(exc))
        finally:
            slot.editor_process = None
            slot.editor_audio_queue = None
            if state_file and os.path.exists(state_file):
                try:
                    os.remove(state_file)
                except OSError:
                    pass

    threading.Thread(target=run, daemon=True).start()


class TrackChain:
    """An ordered VST3 chain for one speaker's track."""

    def __init__(self):
        self.slots = []
        self.enabled = True
        self._lock = threading.Lock()

    def add(self, name, path):
        """
        Loads a plugin and appends it to the chain.

        The load happens INSIDE the lock. Loading while the audio callback was
        inside process() put two threads into pedalboard at once and killed the
        whole app with "PyEval_RestoreThread: the GIL is released" - a native
        crash that no try/except can catch. Every plugin loads fine on its own;
        it was purely the concurrency. Callers should also pause playback, which
        keeps the audio thread out of here entirely.

        Loading itself must also happen on whichever thread first created
        JUCE's MessageManager - the main/Tk thread, since the UI's "add
        effect" path always ran here (see the module comment above
        _load_plugin_on_main_thread). A caller off the main thread - e.g.
        voice_activity.denoise(), which loads rnnoise from the background
        analysis thread - hit "must be reloaded on the main thread" here
        because this used to call pedalboard.load_plugin() directly instead
        of hopping like snapshot() does.

        The load deliberately happens OUTSIDE self._lock. _load_plugin_on_
        main_thread takes the load gate itself now, and process_slots takes
        the gate before this lock - so holding this lock while waiting for the
        gate would invert that order and deadlock against a render in flight.
        Only the append needs the lock.
        """
        plugin = _load_plugin_on_main_thread(path)
        with self._lock:
            self.slots.append(PluginSlot(name, path, plugin))
        return self.slots[-1]

    def add_native(self, key, params=None):
        """Appends a built-in effect. No plugin loading, so no load gate."""
        with self._lock:
            self.slots.append(NativeSlot(key, params))
        return self.slots[-1]

    def remove(self, index):
        with self._lock:
            if 0 <= index < len(self.slots):
                del self.slots[index]

    def move(self, index, delta):
        with self._lock:
            new_index = index + delta
            if 0 <= index < len(self.slots) and 0 <= new_index < len(self.slots):
                self.slots[index], self.slots[new_index] = (
                    self.slots[new_index], self.slots[index])
                return new_index
        return index

    def active_slots(self):
        if not self.enabled:
            return []
        return [s for s in self.slots if not s.bypassed]

    def fingerprint(self):
        """
        Cheap identity for "has anything this chain would render actually
        changed" - same pattern fx_dialog._apply_done already uses to detect
        a chain edited mid-render. `slot.revision` only moves on mark_dirty()
        (a real param/bypass/state change), not on being read or baked, so
        two calls to this between edits compare equal without touching any
        audio. Callers use it to skip reprocessing a whole track when the
        chain that would produce the result hasn't moved since last time.
        """
        return (self.enabled,
                tuple((id(slot), slot.revision) for slot in self.slots))

    def describe(self):
        """
        A one-line summary of the chain for the log: bypassed slots in
        brackets, and a note when the whole chain is switched off.

        Lives here, on the class, and is easy to lose: it spent a while
        orphaned inside _isolated_render_worker, where every caller
        (audio_export, the FX dialog, the chain summary in the log) raised
        AttributeError instead. tests/test_vst_chain.py pins it down.
        """
        if not self.slots:
            return "no plugins"
        parts = [("[" + s.name + "]") if s.bypassed else s.name for s in self.slots]
        return " -> ".join(parts) + ("" if self.enabled else "  (chain off)")

    def process(self, audio, sample_rate, reset=False, log=None,
               should_cancel=None):
        """
        Runs mono float32 `audio` (1-D) through the whole chain in ONE pass.
        See process_slots() below for the mechanics and the reasoning behind
        doing this in one pass rather than in blocks.
        """
        return self.process_slots(audio, sample_rate, self.slots, reset, log,
                                  should_cancel=should_cancel)

    def process_slots(self, audio, sample_rate, slots, reset=False, log=None,
                      gate_timeout=_PROCESS_GATE_TIMEOUT_SECONDS,
                      should_cancel=None):
        """
        Runs mono float32 `audio` (1-D) through exactly `slots` (bypassed
        entries among them are skipped, and the whole call is a no-op if this
        chain is switched off) in ONE pass. Returns the processed array; on
        any plugin error the input is passed through untouched rather than
        dropping audio out.

        Split out from process() so a caller can run only PART of a chain -
        Player uses this to live-process just the slots that have not yet
        been baked into a Track's processed_samples (see player.py's
        pending_slots), and the "Apply to whole track" render in fx_dialog.py
        uses it to render only the newly added slots on top of whatever was
        already baked, instead of redoing the whole chain from raw audio
        every time.

        Deliberately not split into blocks, and the reason is purely about
        correctness: feeding these plugins in chunks changes their output.
        The latency-compensating ones (De-Space, Soap Voice Cleaner, the TDR
        pair) come back misaligned at every block boundary, which would make
        the export sound different from the single pass.

        It is NOT because of the GIL. This used to say pedalboard holds the
        GIL for the length of the call, so the pass froze the app anyway and
        chunking bought nothing. Measured 2026-09-12 and that is false:
        during a 1.22s pass over 5 minutes of audio through Soap Voice
        Cleaner, a counter thread ran MORE iterations than it managed over
        the same wall time with the GIL free. pedalboard releases it, for
        built-in effects and real VST3s alike. So a long offline pass does
        not by itself stop the Tk event loop, and any freeze seen during one
        has some other cause - do not let that old claim end an
        investigation early. Callers still keep offline work off playback,
        but for the plugin-reentrancy reason in snapshot(), not this one.

        A plugin that cannot manage a whole episode in one call (an hour is
        ~600 MB of float32, and plugins declare a maximum block size) falls
        back to chunks for that plugin only, rather than being skipped. Being
        skipped is what used to happen, silently: the waveform simply did not
        change and an export quietly came out with none of the effects on it.

        `log` is how any of that gets said out loud. Playback passes nothing -
        the audio callback must not log per block. `gate_timeout` lets realtime
        playback decline a block immediately when a plugin load is pending;
        offline callers retain the normal bounded wait.

        `should_cancel`, if given, is checked once per plugin - between
        whole-track passes, never mid-plugin-call. A single plugin's call
        over an hour of audio is one opaque native call with no safe
        interruption point (see the chunking note above: splitting it up
        purely to poll a flag would reintroduce the same latency-compensation
        misalignment chunking already exists to avoid). Audio already
        processed by earlier slots when cancellation is observed is returned
        as-is rather than discarded.
        """
        if not self.enabled:
            return audio
        slots = [s for s in slots if not s.bypassed]
        if not slots:
            return audio
        buf = audio.reshape(1, -1)
        with _GATE.processing(timeout=gate_timeout), self._lock:
            for slot in slots:
                if should_cancel and should_cancel():
                    break
                try:
                    with slot.lock:
                        if getattr(slot, "is_native", False):
                            # Built-in effects work on the plain 1-D signal.
                            buf = slot.process(
                                buf.reshape(-1), sample_rate,
                                reset=reset).reshape(1, -1)
                        else:
                            buf = self._run_plugin(slot, buf, sample_rate,
                                                   reset, log)
                except Exception as exc:
                    if log:
                        log(f"  {slot.name}: could not process this audio "
                            f"({exc}) - it is NOT applied here")
                    continue
        return buf.reshape(-1)

    @staticmethod
    def _run_plugin(slot, buf, sample_rate, reset, log):
        """
        One plugin over `buf`, whole if it can manage it and in chunks if it
        cannot. Any length change (latency compensation) is corrected here, so
        a caller never has to decide whether to throw the result away.
        """
        length = buf.shape[1]
        try:
            out = slot.plugin(buf, sample_rate, reset=reset)
        except Exception as exc:
            if log:
                log(f"  {slot.name}: {exc}; retrying in chunks")
            step = int(sample_rate * CHUNK_SECONDS)
            # reset only on the first chunk: the plugin's state has to carry
            # across the joins or every boundary becomes a click. That same
            # statefulness is why the output strategy below is decided ONCE,
            # from chunk 0 only, and never switched mid-loop: if a later
            # chunk came back an unexpected shape, discarding a pre-allocated
            # buffer and "retrying" the whole plugin from chunk 0 in list
            # mode would call process() a second time on a plugin whose
            # internal envelope/compressor/gate state has already advanced
            # once - producing silently different audio, not a safe retry.
            out = None
            for index, offset in enumerate(range(0, length, step)):
                piece = buf[:, offset:offset + step]
                result = slot.plugin(piece, sample_rate,
                                     reset=(reset and index == 0))
                if index == 0:
                    if result.shape[1] != piece.shape[1]:
                        raise RuntimeError(
                            f"{slot.name}: chunked processing returned "
                            f"{result.shape[1]} samples for a "
                            f"{piece.shape[1]}-sample chunk; cannot safely "
                            f"retry once the plugin's internal state has "
                            f"already advanced")
                    # Pre-allocate once chunk 0's shape is confirmed, instead
                    # of building a list and numpy.concatenate-ing a second
                    # full-length array at the end (real, avoidable memory
                    # duplication on long tracks).
                    out = numpy.empty_like(buf)
                out[:, offset:offset + result.shape[1]] = result
            if out is None:
                out = buf
            if log:
                log(f"  {slot.name}: applied in chunks")

        if out.shape[1] != length:
            # Latency-compensating plugins hand back a different length. Line
            # it back up instead of discarding the whole pass, which is what
            # the callers used to do - silently.
            if log:
                log(f"  {slot.name}: returned {out.shape[1]} samples for "
                    f"{length}; aligning")
            if out.shape[1] > length:
                out = out[:, :length]
            else:
                out = numpy.pad(out, ((0, 0), (0, length - out.shape[1])))
        return out

    def snapshot(self, log=None):
        """
        A detached copy of this chain with its own plugin instances.

        Offline work - redrawing the waveform, rendering an export - must never
        touch the live plugins. Those are owned by the audio callback, and
        driving the same VST from two threads (or making the callback wait on a
        full-episode pass) crashes the process outright: pedalboard reports
        "PyEval_RestoreThread: the GIL is released" and takes the app with it.

        Loading fresh instances costs a second or two, which is nothing next to
        the work these callers are about to do anyway.
        """
        copy = TrackChain()
        copy.enabled = self.enabled
        copy.slots = self._snapshot_slot_list(self.slots, log=log)
        return copy

    def _snapshot_slot_list(self, slots, log=None):
        """
        Detached copies of `slots` (bypassed entries skipped), each safe to
        process on a thread other than the one owning the live plugin
        instances - see snapshot()'s docstring for why that matters.

        Shared by snapshot() (copies the whole chain) and the incremental
        "Apply to whole track" render (copies only the not-yet-baked tail,
        see fx_dialog.py), so a re-Apply after appending one more effect only
        has to load/copy that new plugin, not the ones already baked.
        """
        # Phase 1: read each plugin's state. slot.lock is taken and released
        # here and nowhere near the load lock - holding it while waiting for
        # the load lock would invert the order used by process() and deadlock.
        resolved_slots = []
        wanted = []
        for orig_idx, slot in enumerate(list(slots)):
            if slot.bypassed:
                continue                      # nothing to reproduce
            if getattr(slot, "is_native", False):
                # Nothing to load and nothing shared - just take a copy and
                # keep its position in the chain.
                resolved_slots.append((orig_idx, slot.copy()))
                continue
            try:
                with slot.lock:
                    state = slot.plugin.raw_state
            except Exception as exc:
                # One plugin's state failing to read must not take the rest
                # of the chain down with it - snapshot() feeds this into
                # whole-track exports and waveform redraws, and a raise here
                # used to abort the entire render over a single bad plugin.
                if log:
                    log(f"{slot.name}: could not read state ({exc})")
                continue
            wanted.append((orig_idx, slot, state))

        # Phase 2: load the copies. One lock acquisition per plugin rather than
        # one for the whole chain, so a playing audio thread waits for a
        # single load at worst instead of the entire set.
        #
        # Deliberately NOT reused from a pool keyed by slot, even though that
        # would skip the reload cost on every redraw after the first. Tried
        # 2026-09-12 and reverted the same day: it would have meant the same
        # loaded VST3 instance getting process() called on it from a
        # different background thread on every subsequent redraw (each
        # redraw spawns a fresh threading.Thread), never proven safe for a
        # real VST3 the way it was proven for pedalboard releasing the GIL -
        # only exercised there against a synthetic fake plugin with no native
        # thread-affinity behaviour to violate. Not worth the risk on a
        # symptom this project has already been burned by (native crashes
        # from cross-thread pedalboard misuse) without a way to measure it
        # here against the user's actual plugins.
        for orig_idx, slot, state in wanted:
            try:
                plugin = _load_plugin_on_main_thread(slot.path, log=log)
                plugin.raw_state = state
            except Exception as exc:
                if log:
                    log(f"{slot.name}: could not copy for offline use ({exc})")
                continue
            resolved_slots.append((orig_idx, PluginSlot(slot.name, slot.path, plugin)))

        resolved_slots.sort(key=lambda s: s[0])
        return [slot for _, slot in resolved_slots]


def is_available():
    try:
        import pedalboard  # noqa: F401
        return True
    except ImportError:
        return False
