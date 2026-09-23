"""
Spacebar must always toggle play/pause, even when a ttk Button/Checkbutton/
Scale currently has keyboard focus.

Tk's bindtag order for a focused widget is widget-instance, then widget-
CLASS, then toplevel, then "all" - so a plain root.bind("<space>", ...) (the
toplevel tag) only ever runs AFTER a focused button's own class-level <space>
binding (its default "activate the focused widget" behaviour) has already
fired. That let spacebar re-trigger whichever mute/solo/fx/fader control last
had focus instead of toggling playback. The fix overrides the class binding
directly (bind_class, replacing rather than adding to the default) so the
app's own handler runs instead of - not alongside - the widget's own action.

Needs a real Tk root (this is a Windows desktop app; skipped where Tk can't
open a window at all, e.g. a headless CI runner with no display).
"""

import tkinter as tk
from tkinter import ttk

import pytest


@pytest.fixture
def root():
    """
    A real, mapped Tk window, positioned off-screen rather than withdrawn.
    withdraw() prevents the window manager from ever giving it focus, which
    silently breaks focus_set()/event_generate()'s focus-dependent dispatch -
    ttk's own <space> class binding (`ttk::button::activate %W`) only fires
    once a widget genuinely holds focus, which an unmapped window can't grant.
    """
    try:
        r = tk.Tk()
        r.geometry("1x1+2000+2000")
        r.update()
    except tk.TclError as exc:
        pytest.skip(f"no display available for Tk: {exc}")
    yield r
    r.destroy()


def _ensure_focus(root, widget, attempts=10):
    """
    Windows throttles SetForegroundWindow (what focus_force maps to) from a
    process that isn't already the foreground application - exactly the
    situation an automated test run is usually in - so a single request can
    silently be denied. Retries briefly rather than asserting on the first
    attempt; skips (not fails) if the OS never grants it, since that's an
    environment limitation, not a defect in the binding logic under test.
    """
    for _ in range(attempts):
        widget.focus_force()
        root.update()
        if root.focus_get() is widget:
            return
        root.after(20)
        root.update()
    pytest.skip("OS would not grant this process real window focus "
               "(focus-stealing prevention) - cannot exercise focus-"
               "dependent Tk dispatch here")


def test_bind_class_override_fires_instead_of_button_default(root):
    """Mirrors app.py's __init__: override TButton's <space> class binding,
    then confirm pressing space on a focused button runs OUR handler and
    does not also invoke the button."""
    invoked = []
    space_handled = []

    button = ttk.Button(root, text="Mute", command=lambda: invoked.append("button"))
    button.pack()
    _ensure_focus(root, button)

    def on_space(event):
        space_handled.append(event.widget)
        return "break"

    root.bind_class("TButton", "<space>", on_space)
    root.update()

    button.event_generate("<space>")
    root.update()

    assert space_handled == [button]
    assert invoked == [], (
        "the button's own default space-activation fired alongside the "
        "override instead of being replaced by it")


def test_toplevel_bind_still_handles_space_with_no_widget_focus(root):
    """When nothing (or a non-overridden widget) has focus, the plain
    toplevel-level binding is what fires - both are needed together."""
    handled = []
    root.bind("<space>", lambda event: handled.append(event.widget) or "break")
    _ensure_focus(root, root)

    root.event_generate("<space>")
    root.update()

    assert handled == [root]
