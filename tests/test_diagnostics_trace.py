"""
The freeze trace and its watchdog.

Every case here is a bug that was actually shipped into the working tree and
caught in review, not a hypothetical.
"""

import os
import time

import pytest

import diagnostics


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "FREEZE_LOG",
                        str(tmp_path / "autocut_freeze.log"))
    diagnostics._watchdog_stop.clear()
    yield
    diagnostics._watchdog_stop.clear()


# ---------------------------------------------------------------- privacy

def test_strip_paths_reduces_windows_paths_to_basenames():
    line = r'  File "C:\Users\jane\Documents\app\diagnostics.py", line 120 in watch'
    assert diagnostics._strip_paths(line) == (
        '  File "diagnostics.py", line 120 in watch')


def test_strip_paths_reduces_posix_paths_to_basenames():
    line = '  File "/home/jane/app/player.py", line 3 in main'
    assert diagnostics._strip_paths(line) == (
        '  File "player.py", line 3 in main')


def test_strip_paths_leaves_ordinary_text_alone():
    for line in ("Windows fatal exception: int divide by zero",
                 "09:25:40 apply: starting, 178000000 samples, 2 slot(s)",
                 "full_pass: a whole-track render held the lock 41.2s"):
        assert diagnostics._strip_paths(line) == line


def test_strip_paths_leaves_a_bare_slash_alone():
    """
    A slash in prose is not a path. The first version of this rewrote
    "NVIDIA/Intel" to "NVIDIAIntel" and "n/a" to "na" - a diagnostic silently
    corrupting the evidence it exists to preserve.
    """
    for line in ("GPU: NVIDIA/Intel hybrid",
                 "sample rate 44100, n/a channels",
                 "ratio 3/4 applied"):
        assert diagnostics._strip_paths(line) == line


def test_strip_paths_handles_a_unc_path():
    """A network path leaks a server name as readily as a home directory."""
    sep = chr(92)
    unc = sep * 2 + "fileserver" + sep + "share" + sep + "x.py"
    line = '  File "' + unc + '", line 1'

    cleaned = diagnostics._strip_paths(line)

    assert cleaned == '  File "x.py", line 1'
    assert "fileserver" not in cleaned

def test_report_carries_no_home_directory_path():
    """
    The module's contract is basenames only - the report is written to be
    emailed to a stranger. Adding the freeze log to it leaked the absolute
    paths inside faulthandler dumps.
    """
    diagnostics.trace("something")
    with open(diagnostics.FREEZE_LOG, "a", encoding="utf-8") as handle:
        handle.write(r'  File "C:\Users\jane\secret-client\app.py", line 1' + "\n")

    text = diagnostics.summary(description="test")
    assert "secret-client" not in text
    assert r"C:\Users\jane" not in text


# ---------------------------------------------------------------- rotation

def test_oversized_log_is_trimmed_to_its_tail(monkeypatch):
    monkeypatch.setattr(diagnostics, "FREEZE_LOG_MAX_BYTES", 4096)
    with open(diagnostics.FREEZE_LOG, "w", encoding="utf-8") as handle:
        handle.write("x" * 20000)
        handle.write("THE-RECENT-PART")

    diagnostics._rotate_if_large()

    size = os.path.getsize(diagnostics.FREEZE_LOG)
    assert size < 4096
    with open(diagnostics.FREEZE_LOG, encoding="utf-8") as handle:
        kept = handle.read()
    assert "THE-RECENT-PART" in kept
    assert "trimmed" in kept


def test_rotation_leaves_a_small_log_alone():
    diagnostics.trace("a line worth keeping")
    before = open(diagnostics.FREEZE_LOG, encoding="utf-8").read()
    diagnostics._rotate_if_large()
    assert open(diagnostics.FREEZE_LOG, encoding="utf-8").read() == before


# ---------------------------------------------------------------- watchdog

@pytest.fixture(scope="module")
def tk_root():
    """
    ONE root for the whole module, reused.

    A fresh tk.Tk() per test fails once an earlier root has been destroyed in
    the same process, and a broad "skip if it raises" hid that as a phantom
    "no display available" - so a test that should have run simply vanished
    from the results while the suite still read green. mainloop() can be
    entered and left repeatedly on a single root, so reuse is enough.
    """
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:              # genuinely headless
        pytest.skip(f"no display available: {exc}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


def test_a_clean_shutdown_logs_no_stall(tk_root):
    """
    The heartbeat stops when mainloop() returns. A watchdog left running then
    calls that a freeze, so every normal exit wrote a fake stall into the one
    file someone reads to find a real one.
    """
    root = tk_root
    diagnostics.start_event_loop_watchdog(root, stall_seconds=0.5,
                                          poll_seconds=0.1)
    root.after(300, root.quit)
    root.mainloop()
    diagnostics.stop_event_loop_watchdog()

    time.sleep(1.5)                          # well past stall_seconds
    # No file at all is the ideal outcome - nothing had anything to say.
    log = ""
    if os.path.exists(diagnostics.FREEZE_LOG):
        log = open(diagnostics.FREEZE_LOG, encoding="utf-8").read()
    assert "STALLED" not in log


def test_a_real_stall_is_still_caught(tk_root):
    """The stop flag must not have disarmed the thing entirely."""
    root = tk_root
    diagnostics.start_event_loop_watchdog(root, stall_seconds=0.5,
                                          poll_seconds=0.1)
    root.after(200, lambda: time.sleep(2.0))
    root.after(2600, root.quit)
    root.mainloop()
    diagnostics.stop_event_loop_watchdog()

    log = open(diagnostics.FREEZE_LOG, encoding="utf-8").read()
    assert "EVENT LOOP STALLED" in log
    assert "Thread" in log                   # the all-threads dump landed
