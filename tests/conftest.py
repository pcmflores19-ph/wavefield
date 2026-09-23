import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "auto_cut"))

import pytest


@pytest.fixture(autouse=True)
def _isolate_freeze_log(tmp_path, monkeypatch):
    """
    Every test run was writing straight into the real auto_cut/autocut_freeze.log
    - diagnostics.FREEZE_LOG is a fixed path next to diagnostics.py, not test-
    isolated, and test_vst_gate.py deliberately exercises the "main thread never
    loaded it" hang path (see test_abandoned_load_never_fires) with a plugin
    named "orphan.vst3" to verify the timeout logic. Confirmed 2026-09-12: those
    synthetic entries land in the SAME file the user was told (FREEZE-HANDOFF.md)
    to read first for evidence of a real freeze, indistinguishable from a live
    recurrence - a suite run while diagnosing a real report corrupts the one
    piece of evidence that report depends on. Every test now gets its own log
    file instead.
    """
    import diagnostics
    monkeypatch.setattr(diagnostics, "FREEZE_LOG",
                        str(tmp_path / "test_autocut_freeze.log"))


@pytest.fixture(autouse=True)
def _isolate_cache_dir(tmp_path, monkeypatch):
    """
    settings.cache_dir() points at the real LOCALAPPDATA cache - waveform.py's
    on-disk peaks cache and voice_activity.py's VAD cache both write there
    keyed on nothing test-specific but path/size/mtime, so a suite run
    without this would read back a real cached array left over from actual
    app use (or write test fixture noise into it). Every test gets its own
    empty directory instead.
    """
    import settings
    cache_path = tmp_path / "test_cache"
    monkeypatch.setattr(settings, "cache_dir", lambda: str(cache_path))
