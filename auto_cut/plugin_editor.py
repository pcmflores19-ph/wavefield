"""
Opens one VST3 plugin's native editor, in its own process.

pedalboard refuses to show a plugin UI from anything but the main thread, and
the call blocks until the window closes. Calling it directly from the app would
freeze the whole interface - meters, playhead and all - for as long as the
editor is open. So the editor runs here instead, as a subprocess:

    parent  --(plugin path + current raw_state)-->  this script
    this script  shows the editor on ITS main thread, user tweaks, closes
    parent  <--(new raw_state on stdout)--  this script
    parent  applies that state to its own live plugin instance

Run directly:  python plugin_editor.py <plugin path> [<state file>]
Prints the resulting state as base64 on stdout, prefixed with STATE:.
"""

import base64
import os
import struct
import sys
import threading
import time

import numpy as np

# How often the plugin's state is checked while its editor is open. Fast enough
# that a knob move is audible almost immediately, slow enough not to thrash.
STATE_POLL_SECONDS = 0.15


# How often the editor's always-on-top standing is CHECKED. The plugin window
# belongs to THIS process, while the main app is a different one, so nothing
# here gets told when the app raises itself - polling is the only way to notice.
#
# Checking is not the same as re-applying, and the difference is a bug. This
# used to call SetWindowPos unconditionally every second, which is what made
# the Soap Voice Cleaner preset menu unusable: a JUCE dropdown is its own
# top-level window, and re-raising the editor into the top of the topmost band
# put the editor back in front of the open menu about a second after it opened.
# You could see the list, but never click an entry.
#
# Measured on Soap Voice Cleaner: the topmost flag survives for as long as the
# editor is open, so the re-apply was doing nothing useful even when it was not
# doing harm. It now fires only when the flag has actually been lost.
TOPMOST_REASSERT_SECONDS = 1.0

# Editor windows are big; a dropdown or tooltip is small and transient. Any
# EXTRA visible window from this process means something is open in front of
# the editor, and the z-order must not be touched until it closes.
WS_EX_TOPMOST = 0x00000008
GWL_EXSTYLE = -20


def _center_editor_window(title, timeout=15.0):
    """
    JUCE opens the plugin editor at roughly (-8, -31) - its title bar sits above
    the top of the screen, so the window can't be dragged and its close button
    is unreachable. Wait for the window to appear, then move it on-screen, give
    it a useful title, and keep it in front of the main window.

    In front matters because the editor is a SEPARATE PROCESS from the app.
    Pressing Play raises the app's own window, which buried the plugin editor
    behind it with no way back except the taskbar. Marking the editor topmost
    keeps it above the app while you tune a plugin and listen.

    Runs on a worker thread because show_editor() owns the main thread.
    """
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    # argtypes are NOT optional here. HWND is a 64-bit pointer, and
    # HWND_TOPMOST is the sentinel (HWND)-1. Left undeclared, ctypes marshals
    # the Python int -1 as a 32-bit C int, so SetWindowPos receives 0xFFFFFFFF
    # instead of 0xFFFFFFFFFFFFFFFF, rejects it as an invalid window, and
    # silently does nothing - which is exactly how the first attempt at this
    # failed: the editor was positioned but never actually raised.
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.GetWindowRect.argtypes = [wintypes.HWND,
                                     ctypes.POINTER(wintypes.RECT)]

    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    pid = os.getpid()

    SWP_NOSIZE, SWP_SHOWWINDOW = 0x0001, 0x0040
    SWP_NOMOVE, SWP_NOACTIVATE = 0x0002, 0x0010
    # Built as a real HWND, so the value that reaches the API is the
    # full-width (HWND)-1 the docs call for.
    HWND_TOPMOST = wintypes.HWND(-1)

    def window_area(hwnd):
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return 0
        return max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)

    def juce_windows():
        """Every visible JUCE window this process owns, topmost-first."""
        found = []

        def callback(hwnd, _lparam):
            window_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
            if window_pid.value == pid and user32.IsWindowVisible(hwnd):
                cls = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, cls, 256)
                if cls.value.startswith("JUCE"):
                    found.append(hwnd)
            return True

        user32.EnumWindows(enum_proc(callback), 0)
        return found

    def place(hwnd):
        """Move on-screen, name it, raise it. Once per window."""
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width = rect.right - rect.left
        height = rect.bottom - rect.top

        # Doubling this via SetWindowPos was tried and measured directly:
        # JUCE manages its own top-level window size and reverts an external
        # resize within milliseconds (native size and "doubled" size measured
        # identical, ~330x536, in a controlled test). The plugin's editor
        # size is not controllable from here - only position/z-order are.
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 3)

        # ONE call, not hide-then-show. The previous version hid the window
        # first (SWP_HIDEWINDOW) to avoid a visible jump from JUCE's default
        # spawn position (roughly (-8, -31), title bar above the screen) to
        # centered - but pedalboard/JUCE treats that hide as the window being
        # CLOSED: show_editor() returned immediately, tearing the editor down
        # before the second call ever ran. Reproduced directly, isolated from
        # everything else in this file (a standalone script doing only the
        # hide call, nothing else, made show_editor() return in ~1.5s instead
        # of blocking) - this is what made every plugin's editor flash open
        # and vanish. A visible jump to center is a real but far smaller cost
        # than the editor never working at all.
        #
        # SWP_NOZORDER is deliberately NOT passed: it is what kept this call
        # from changing the z-order, so the editor was merely placed, never
        # raised, and the app's next Play buried it.
        #
        user32.SetWindowPos(hwnd, HWND_TOPMOST, x, y, 0, 0,
                            SWP_NOSIZE | SWP_SHOWWINDOW)
        user32.SetWindowTextW(hwnd, title)
        user32.SetForegroundWindow(hwnd)

    hwnd = None
    appear_deadline = time.time() + timeout
    # How long to keep watching after the tracked window disappears. Several
    # plugins destroy and rebuild their editor on a preset change, and the
    # rebuilt window arrives back at JUCE's off-screen default with no topmost
    # flag. The old code returned the moment the window went away, so the first
    # preset change left the editor unreachable again.
    rebuild_grace = 10.0
    gone_since = None

    while True:
        windows = juce_windows()

        if hwnd is None or not user32.IsWindow(hwnd):
            if not windows:
                now = time.time()
                if hwnd is None:
                    if now > appear_deadline:
                        return                  # the editor never opened
                    # Tighter than the 0.25s used below while waiting out a
                    # rebuild: every tick here is extra time the window can
                    # sit visible at JUCE's off-screen default before place()
                    # ever runs, on top of the hide-then-move in place()
                    # itself. Only matters for a few hundred ms around
                    # startup, so the tighter loop costs nothing noticeable.
                    time.sleep(0.02)
                    continue
                else:
                    gone_since = gone_since or now
                    if now - gone_since > rebuild_grace:
                        return                  # closed for good
                time.sleep(0.25)
                continue
            # The largest, not the topmost. Both the editor and a dropdown are
            # JUCE windows of this process, and after a rebuild the scan can
            # land while a menu is open - adopting that would move and retitle
            # the menu instead of the editor.
            hwnd = max(windows, key=window_area)
            gone_since = None
            place(hwnd)
            time.sleep(TOPMOST_REASSERT_SECONDS)
            continue

        # A second visible window means a dropdown, tooltip or dialog is open in
        # front of the editor. Touching the z-order now is what made the preset
        # menu unusable, so leave it strictly alone until it closes.
        if len(windows) > 1:
            time.sleep(TOPMOST_REASSERT_SECONDS)
            continue

        # Re-raise only if the flag was genuinely lost. On a healthy editor this
        # is never true, so the steady state makes no SetWindowPos calls at all.
        exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if not exstyle & WS_EX_TOPMOST:
            # Without activating, so this never steals focus from whatever the
            # user is actually typing into.
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
        time.sleep(TOPMOST_REASSERT_SECONDS)


def _make_dpi_aware():
    """
    Matches app.py's own _make_dpi_aware(). This runs as a SEPARATE process
    (see the module docstring) and never inherits the main app's declaration -
    Windows DPI awareness is set per-process, not per-window. Without this,
    the main app renders at real screen resolution while this process gets
    DPI-virtualized (rendered at 96 DPI, then bitmap-stretched by Windows) -
    on a scaled display the two end up at different effective sizes, which
    is what made the plugin editor read as smaller than the rest of the app
    after app.py gained its own DPI awareness. Must run before any window
    (the plugin's editor included) is created.
    """
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main():
    _make_dpi_aware()

    if len(sys.argv) < 2:
        print("usage: plugin_editor.py <plugin path> [<state file>]", file=sys.stderr)
        return 2

    plugin_path = sys.argv[1]
    state_path = sys.argv[2] if len(sys.argv) > 2 else None

    import pedalboard

    plugin = pedalboard.load_plugin(plugin_path)

    if state_path:
        try:
            with open(state_path, "rb") as f:
                data = f.read()
            if data:
                plugin.raw_state = data
        except Exception as exc:
            print(f"could not restore plugin state: {exc}", file=sys.stderr)

    # Nudge the window on-screen once JUCE has created it. Must happen on a
    # worker thread - show_editor() takes over the main thread below.
    title = os.path.splitext(os.path.basename(plugin_path.rstrip("\\/")))[0]
    threading.Thread(target=_center_editor_window, args=(title,), daemon=True).start()

    # Stream state out while the editor is open, so the parent can apply each
    # tweak to its live plugin and you hear the change as you make it, rather
    # than only once the window closes.
    stop_watching = threading.Event()

    # Guards every direct touch of `plugin` from a thread OTHER than the main
    # thread (which owns show_editor()'s native message loop): the state
    # read below, and the audio feed's process() call. Confirmed directly
    # (see the project's diagnosis) that pedalboard/JUCE tolerates
    # process() running concurrently with an open show_editor() on the same
    # instance - this lock is about not overlapping OUR OWN two call sites,
    # not a workaround for that.
    plugin_lock = threading.Lock()

    def emit_state():
        try:
            with plugin_lock:
                return base64.b64encode(plugin.raw_state).decode("ascii")
        except Exception:
            return None

    def watch_state():
        last = emit_state()
        while not stop_watching.is_set():
            time.sleep(STATE_POLL_SECONDS)
            current = emit_state()
            if current and current != last:
                last = current
                print("STATE:" + current, flush=True)

    threading.Thread(target=watch_state, daemon=True).start()

    # Feed this plugin instance the same audio the app is actually playing
    # (see vst_host._forward_editor_audio / player.py's tap in _mix_into),
    # purely so a live meter/gain-reduction display in the plugin's own GUI
    # has real signal to draw - the output is discarded, this instance is
    # never the one that's actually heard. Silently stops on EOF (parent
    # closed stdin) or any plugin error; a dead meter is not worth crashing
    # the editor over.
    sample_rate = int(os.environ.get("AUTOCUT_EDITOR_SAMPLE_RATE", "48000"))

    def _read_exact(n):
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sys.stdin.buffer.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def feed_audio():
        first = True
        while True:
            header = _read_exact(4)
            if header is None:
                return
            (count,) = struct.unpack("<I", header)
            payload = _read_exact(count * 4)
            if payload is None:
                return
            try:
                buf = np.frombuffer(payload, dtype=np.float32).reshape(1, -1)
                with plugin_lock:
                    plugin.process(buf, sample_rate, reset=first)
                first = False
            except Exception:
                return

    threading.Thread(target=feed_audio, daemon=True).start()

    plugin.show_editor()      # blocks here until the user closes the window
    stop_watching.set()

    final = emit_state()
    if final is None:
        print("could not read back plugin state", file=sys.stderr)
        return 1
    print("STATE:" + final, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
