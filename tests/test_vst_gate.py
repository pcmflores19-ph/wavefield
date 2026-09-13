"""
The plugin-load gate protocol in vst_host.

These cover a native crash that no try/except can catch, so they test the
protocol rather than the plugins: pedalboard is faked, and what's asserted is
that two loads never overlap and that nothing loads after its requester gave
up. See autocut_crash.log for the failure these came from - the main thread
inside load_plugin while a worker was inside _run_plugin.
"""

import queue
import sys
import threading
import time
import types

import pytest


@pytest.fixture
def gate(monkeypatch):
    """vst_host with pedalboard faked out and a stand-in Tk main loop."""
    state = types.SimpleNamespace(loads=[], overlaps=[], active=0,
                                  lock=threading.Lock())

    def fake_load(path):
        with state.lock:
            state.active += 1
            if state.active > 1:
                state.overlaps.append(path)
        time.sleep(0.02)
        with state.lock:
            state.active -= 1
            state.loads.append(path)
        return object()

    fake = types.ModuleType("pedalboard")
    fake.load_plugin = fake_load
    monkeypatch.setitem(sys.modules, "pedalboard", fake)

    import vst_host

    state.queue = queue.Queue()
    monkeypatch.setattr(vst_host, "_main_thread_runner", state.queue.put)
    # A fresh gate per test, so one test's leftovers cannot block the next.
    monkeypatch.setattr(vst_host, "_GATE", vst_host._ProcessLoadGate())

    def pump(timeout=2.0):
        """Run whatever the worker queued, the way the Tk main loop would."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state.queue.get(timeout=0.01)()
            except queue.Empty:
                return

    state.pump = pump
    state.module = vst_host
    return state


def test_hop_load_completes(gate):
    done = []
    thread = threading.Thread(
        target=lambda: done.append(
            gate.module._load_plugin_on_main_thread("worker.vst3")),
        daemon=True)
    thread.start()
    gate.pump()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert done and gate.loads == ["worker.vst3"]


def test_main_thread_add_does_not_deadlock_against_a_hop(gate):
    """
    The crash-log sequence: a worker asks for a hop load at the moment the
    main thread runs TrackChain.add().

    This used to deadlock for the full 30s timeout - the worker held the gate
    while waiting on the main thread, and the main thread was blocked taking
    the same gate, so it never ran the worker's closure.
    """
    errors = []

    def worker():
        try:
            gate.module._load_plugin_on_main_thread("worker.vst3")
        except Exception as exc:            # noqa: BLE001 - recorded, asserted below
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.05)                        # let the closure get queued

    chain = gate.module.TrackChain()
    started = time.time()
    chain.add("mine", "mine.vst3")          # the main thread's own load
    elapsed = time.time() - started

    gate.pump()
    thread.join(timeout=5)

    assert elapsed < 2.0, "the main thread blocked - the gate inverted again"
    assert not thread.is_alive() and not errors
    assert len(chain.slots) == 1
    assert sorted(gate.loads) == ["mine.vst3", "worker.vst3"]
    assert gate.overlaps == [], "two loads ran at once - this is the crash"


def test_abandoned_load_never_fires(gate, monkeypatch):
    """
    A hop that times out must not still load later.

    The closure stays queued after the requester gives up; left alone it ran
    on the main thread with no gate held, while the requester was already back
    inside process() - two threads in pedalboard, native access violation.
    """
    monkeypatch.setattr(gate.module, "_MAIN_THREAD_LOAD_TIMEOUT_SECONDS", 0.1)
    errors = []

    def worker():
        try:
            gate.module._load_plugin_on_main_thread("orphan.vst3")
        except Exception as exc:            # noqa: BLE001 - recorded, asserted below
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert errors and isinstance(errors[0], RuntimeError)

    gate.pump()                             # the main thread gets to it at last
    assert gate.loads == [], "an abandoned load fired anyway"


def test_processing_and_loading_never_overlap(gate):
    """
    The gate's core promise, exercised from both sides at once.

    The render loop sleeps OUTSIDE the gate, the way the audio callback leaves
    a gap between blocks. Re-acquiring with no gap starves the load side
    indefinitely - the gate has no fairness - which is why every caller that
    loads stops playback first.
    """
    module = gate.module
    observed = []
    stop = threading.Event()

    def render():
        while not stop.is_set():
            with module._GATE.processing():
                observed.append(gate.active)
            time.sleep(0.005)

    worker = threading.Thread(target=render, daemon=True)
    worker.start()
    for _ in range(3):
        module._load_plugin_on_main_thread("p.vst3")
    stop.set()
    worker.join(timeout=2)

    assert gate.loads == ["p.vst3"] * 3
    assert gate.overlaps == [], "a load overlapped another load"
    # No render ever observed a load in flight.
    assert observed and all(count == 0 for count in observed)
