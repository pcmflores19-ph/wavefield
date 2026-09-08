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
import subprocess
import sys
import tempfile
import threading

import bundled

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
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._processing = 0
        self._loading = False

    @contextlib.contextmanager
    def processing(self):
        with self._cv:
            while self._loading:
                self._cv.wait()
            self._processing += 1
        try:
            yield
        finally:
            with self._cv:
                self._processing -= 1
                self._cv.notify_all()

    @contextlib.contextmanager
    def loading(self):
        with self._cv:
            while self._loading or self._processing:
                self._cv.wait()
            self._loading = True
        try:
            yield
        finally:
            with self._cv:
                self._loading = False
                self._cv.notify_all()


# A thread that is already inside processing() must never try to load: that
# would wait on itself. Every caller here loads first, then processes.
_GATE = _ProcessLoadGate()


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

    def apply_state(self, raw_state):
        with self.lock:
            self.plugin.raw_state = raw_state


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

    def process(self, audio, sample_rate):
        import effects
        return effects.apply(self.key, audio, sample_rate, self.params)

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
                current = slot.plugin.raw_state
            except Exception:
                current = b""

            if current:
                fd, state_file = tempfile.mkstemp(prefix="autocut_vststate_", suffix=".bin")
                with os.fdopen(fd, "wb") as f:
                    f.write(current)

            cmd = bundled.editor_command(slot.path, state_file)

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, bufsize=1)
            slot.editor_process = proc

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
        """
        import pedalboard
        with _GATE.loading(), self._lock:
            plugin = pedalboard.load_plugin(path)
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

    def process(self, audio, sample_rate, reset=False):
        """
        Runs mono float32 `audio` (1-D) through the chain in ONE pass. Returns
        the processed array; on any plugin error the input is passed through
        untouched rather than dropping audio out.

        Deliberately not split into blocks. pedalboard holds the GIL for the
        length of a process() call, so a full-episode pass does freeze the rest
        of the app - but feeding these plugins in chunks changes their output:
        the latency-compensating ones (De-Space, Soap Voice Cleaner, the TDR
        pair) come back misaligned at every block boundary, which would make the
        export sound different from the single pass. Correct audio wins; the
        callers instead make sure offline work never overlaps playback.
        """
        slots = self.active_slots()
        if not slots:
            return audio
        buf = audio.reshape(1, -1)
        with _GATE.processing(), self._lock:
            for slot in slots:
                try:
                    with slot.lock:
                        if getattr(slot, "is_native", False):
                            # Built-in effects work on the plain 1-D signal.
                            buf = slot.process(
                                buf.reshape(-1), sample_rate).reshape(1, -1)
                        else:
                            buf = slot.plugin(buf, sample_rate, reset=reset)
                except Exception:
                    continue
        return buf.reshape(-1)

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
        import pedalboard

        # Phase 1: read each plugin's state. slot.lock is taken and released
        # here and nowhere near the load lock - holding it while waiting for
        # the load lock would invert the order used by process() and deadlock.
        copy = TrackChain()
        copy.enabled = self.enabled

        resolved_slots = []
        wanted = []
        for orig_idx, slot in enumerate(list(self.slots)):
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
                if log:
                    log(f"{slot.name}: could not read state ({exc})")
                continue
            wanted.append((orig_idx, slot.name, slot.path, state))

        # Phase 2: load the copies. One lock acquisition per plugin rather than
        # one for the whole chain, so a playing audio thread waits for a single
        # load at worst instead of the entire set.
        for orig_idx, name, path, state in wanted:
            try:
                with _GATE.loading():
                    plugin = pedalboard.load_plugin(path)
                plugin.raw_state = state
            except Exception as exc:
                if log:
                    log(f"{name}: could not copy for offline use ({exc})")
                continue
            resolved_slots.append((orig_idx, PluginSlot(name, path, plugin)))

        resolved_slots.sort(key=lambda s: s[0])
        copy.slots = [slot for _, slot in resolved_slots]
        return copy

    def describe(self):
        if not self.slots:
            return "no plugins"
        parts = [("[" + s.name + "]") if s.bypassed else s.name for s in self.slots]
        return " -> ".join(parts) + ("" if self.enabled else "  (chain off)")


def is_available():
    try:
        import pedalboard  # noqa: F401
        return True
    except ImportError:
        return False


def platform_note():
    if platform.system() != "Windows":
        return "VST3 search paths are Windows-specific; add your own directories."
    return ""
