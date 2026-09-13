"""
The plugin gate's two anti-freeze properties, both from the 2026-09-12
"Apply to whole track" lock-up: the app went permanently unclickable while
audio carried on playing.

Neither is about plugins, so neither needs pedalboard - they are properties of
_ProcessLoadGate on its own.
"""

import threading
import time

import pytest

import vst_host


@pytest.fixture
def gate():
    return vst_host._ProcessLoadGate()


def test_pending_load_stops_new_processing_slipping_in(gate):
    """
    Writer priority. A steady stream of processing passes - what the audio
    callback is, one every ~23ms for as long as playback runs - must not be
    able to starve a load out of the gate forever.

    Before this, processing() only waited while a load was already under way,
    and a load only became "under way" once it had won, so the count it was
    waiting on never reached zero.
    """
    stop = threading.Event()
    loaded = threading.Event()
    entered_first = threading.Event()

    def keep_processing():
        # Stands in for the audio callback: re-enters the gate continuously.
        first = True
        while not stop.is_set():
            try:
                with gate.processing(timeout=2.0):
                    if first:
                        entered_first.set()
                        first = False
                    time.sleep(0.005)
            except vst_host.GateUnavailable:
                # Correct behaviour once a load is pending: queue, don't
                # barge. Keep trying so the starvation attempt is genuine.
                time.sleep(0.005)

    hammer = threading.Thread(target=keep_processing, daemon=True)
    hammer.start()
    assert entered_first.wait(2.0), "the processing hammer never started"

    def load():
        with gate.loading(timeout=5.0):
            loaded.set()

    loader = threading.Thread(target=load, daemon=True)
    loader.start()

    got_in = loaded.wait(5.0)
    stop.set()
    hammer.join(timeout=2.0)
    loader.join(timeout=2.0)

    assert got_in, "a pending load was starved by continuous processing"


def test_loading_aborts_when_the_caller_gives_up(gate):
    """
    The wait is released by `abort` DURING the wait, not only before it.

    This one is the freeze itself: the worker that queued the load timed out
    and set its cancelled flag, but the load closure was already parked inside
    the gate on the Tk main thread and never looked at the flag again. With no
    event loop left, nothing could close the progress dialog, stop playback,
    or release the modal grab.
    """
    cancelled = threading.Event()
    held = threading.Event()
    release = threading.Event()

    def hold_processing():
        with gate.processing(timeout=5.0):
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold_processing, daemon=True)
    holder.start()
    assert held.wait(2.0), "the processing holder never started"

    # Long timeout on purpose: only `abort` should end this wait.
    started = time.monotonic()
    threading.Timer(0.2, cancelled.set).start()
    with pytest.raises(vst_host.GateUnavailable):
        with gate.loading(timeout=30.0, abort=cancelled.is_set):
            pytest.fail("the gate was taken while processing was still held")
    elapsed = time.monotonic() - started

    release.set()
    holder.join(timeout=2.0)

    assert elapsed < 5.0, (
        f"abort took {elapsed:.1f}s to release the waiter; it must not sit "
        "out the full timeout")


def test_loading_times_out_rather_than_waiting_forever(gate):
    """No wait here is unbounded, because this one can run on the Tk thread."""
    held = threading.Event()
    release = threading.Event()

    def hold_processing():
        with gate.processing(timeout=5.0):
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold_processing, daemon=True)
    holder.start()
    assert held.wait(2.0), "the processing holder never started"

    with pytest.raises(vst_host.GateUnavailable):
        with gate.loading(timeout=0.3):
            pytest.fail("the gate was taken while processing was still held")

    release.set()
    holder.join(timeout=2.0)


def test_gate_is_reusable_after_a_failed_load(gate):
    """
    A load that times out or aborts must leave no trace. If its pending count
    survived, every later processing pass would queue behind a load that is
    never coming, which is the same freeze wearing a different hat.
    """
    held = threading.Event()
    release = threading.Event()

    def hold_processing():
        with gate.processing(timeout=5.0):
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold_processing, daemon=True)
    holder.start()
    assert held.wait(2.0)

    with pytest.raises(vst_host.GateUnavailable):
        with gate.loading(timeout=0.2):
            pass

    release.set()
    holder.join(timeout=2.0)

    # Both operations must work normally again.
    with gate.processing(timeout=1.0):
        pass
    with gate.loading(timeout=1.0):
        pass

    assert gate._load_waiting == 0
    assert gate._processing == 0
    assert gate._loading is False


def test_processing_passes_still_overlap(gate):
    """
    The original reason this is not a plain mutex: the audio callback must not
    wait out an offline pass. Writer priority must not have quietly turned it
    into one.
    """
    both_in = threading.Barrier(2, timeout=2.0)

    def pass_through():
        with gate.processing(timeout=2.0):
            both_in.wait()

    threads = [threading.Thread(target=pass_through, daemon=True)
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)
        assert not t.is_alive(), "two processing passes could not overlap"
