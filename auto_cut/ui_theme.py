"""
DaVinci Resolve-ish dark theme for the tkinter UI.

The exported timeline lands in Resolve, so the tool looks like where the work
ends up rather than like a stock tkinter dialog. Colours are sampled from
Resolve 21's edit page: near-black panels, mid-grey controls, a single warm
accent used sparingly.
"""

import tkinter as tk
from tkinter import ttk

# Palette
BG = "#1c1c1c"            # window background
PANEL = "#252525"         # panel / section background
PANEL_LIGHT = "#2e2e2e"   # raised controls
BORDER = "#3a3a3a"
TIMELINE_BG = "#141414"   # waveform / meter canvases
TEXT = "#d8d8d8"
# #8c8c8c measured ~4-4.6:1 against PANEL/PANEL_LIGHT - right at WCAG AA's
# floor for normal text, which read as illegible in the effects window and
# the toggle buttons next to the menu bar. Brightened for real headroom.
TEXT_DIM = "#a8a8a8"
ACCENT = "#e08a3c"        # Resolve's orange, for the active page and playhead
ACCENT_DIM = "#8a5626"
SELECT = "#2f5d8a"        # selection blue

# Track lane colours, reused by the waveform, mixer swatches and meters.
LANE_COLORS = ["#57b9a6", "#c9a227", "#7a9ec2", "#c07ab8", "#9ec27a"]

FONT = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_TITLE = ("Segoe UI", 11, "bold")


def apply(root):
    """Applies the theme to a Tk root. Returns the ttk.Style for further tweaks."""
    root.configure(background=BG)

    style = ttk.Style(root)
    # 'clam' is the only built-in theme that honours background colours on
    # Windows; 'vista' ignores most of what we set here.
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(".", background=BG, foreground=TEXT, font=FONT,
                    fieldbackground=PANEL_LIGHT, bordercolor=BORDER,
                    lightcolor=PANEL_LIGHT, darkcolor=PANEL)

    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
    style.configure("Dim.TLabel", background=BG, foreground=TEXT_DIM)
    style.configure("PanelDim.TLabel", background=PANEL, foreground=TEXT_DIM)
    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=FONT_TITLE)
    style.configure("Value.TLabel", background=BG, foreground=ACCENT, font=FONT_BOLD)

    # No border at all. clam draws a Labelframe's frame from lightcolor and
    # darkcolor rather than bordercolor, and even matched to the background it
    # still reads as a heavy box on a dark window. The caption above each
    # group separates them perfectly well on its own.
    style.configure("TLabelframe", background=BG, bordercolor=BG,
                    lightcolor=BG, darkcolor=BG,
                    relief="flat", borderwidth=0)
    # A group whose background matches the panel inside it - otherwise the
    # darker window colour shows as a margin around every list and reads as a
    # thick border, which is what it looked like.
    style.configure("Flush.TLabelframe", background=PANEL, bordercolor=PANEL,
                    lightcolor=PANEL, darkcolor=PANEL,
                    relief="flat", borderwidth=0)
    style.configure("Flush.TLabelframe.Label", background=PANEL,
                    foreground=TEXT_DIM, font=FONT_SMALL)
    style.configure("TLabelframe.Label", background=BG, foreground=TEXT_DIM,
                    font=FONT_SMALL)

    style.configure("TButton", background=PANEL_LIGHT, foreground=TEXT,
                    bordercolor=BORDER, focuscolor=BG, padding=(8, 4))
    style.map("TButton",
              background=[("pressed", ACCENT_DIM), ("active", "#3a3a3a"),
                          ("disabled", PANEL)],
              foreground=[("disabled", "#5a5a5a")])

    # The menu bar is our own strip of Menubuttons, not the OS menu bar.
    # Windows draws its own menu bar and ignores tk's colours entirely, which
    # left the labels sitting in system colours over a dark window.
    style.configure("Menubar.TFrame", background=PANEL)
    style.configure("Menu.TMenubutton", background=PANEL, foreground=TEXT,
                    borderwidth=0, relief="flat", padding=(10, 4),
                    arrowsize=0, font=FONT)
    style.map("Menu.TMenubutton",
              background=[("active", PANEL_LIGHT), ("pressed", SELECT)],
              foreground=[("active", "#ffffff"), ("pressed", "#ffffff")])

    style.configure("Accent.TButton", background=ACCENT_DIM, foreground="#f0f0f0")
    style.map("Accent.TButton", background=[("active", ACCENT), ("disabled", PANEL)])

    # A checkbutton that looks like a button rather than a tick box, for
    # Auto-cut and Auto-mute. A Checkbutton rather than a Button because it
    # already owns the on/off variable and gets the "selected" state for free;
    # only the indicator has to go, which is done by cloning the button layout.
    # Off reads as an outline, on as a filled accent - the same colour language
    # as Accent.TButton, so it looks native to the rest of the window.
    try:
        style.layout("Toggle.TCheckbutton", style.layout("TButton"))
    except tk.TclError:
        pass                # falls back to a normal checkbutton, still usable
    style.configure("Toggle.TCheckbutton", background=PANEL_LIGHT,
                    foreground=TEXT_DIM, bordercolor=BORDER,
                    focuscolor=PANEL_LIGHT, padding=(6, 5), anchor="center",
                    font=FONT)
    style.map("Toggle.TCheckbutton",
              background=[("selected", "!disabled", ACCENT_DIM),
                          ("active", "#3a3a3a"),
                          ("disabled", PANEL)],
              foreground=[("selected", "!disabled", "#f8f8f8"),
                          ("disabled", "#5a5a5a")],
              bordercolor=[("selected", ACCENT)],
              lightcolor=[("selected", ACCENT_DIM)],
              darkcolor=[("selected", ACCENT_DIM)])

    style.configure("TCheckbutton", background=BG, foreground=TEXT,
                    focuscolor=BG, indicatorcolor=PANEL_LIGHT)
    style.map("TCheckbutton",
              background=[("active", BG)],
              indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT_DIM)])
    style.configure("Panel.TCheckbutton", background=PANEL, foreground=TEXT)
    style.map("Panel.TCheckbutton", background=[("active", PANEL)])

    style.configure("TRadiobutton", background=BG, foreground=TEXT, focuscolor=BG)
    style.map("TRadiobutton", background=[("active", BG)],
              indicatorcolor=[("selected", ACCENT)])

    style.configure("TEntry", fieldbackground=PANEL_LIGHT, foreground=TEXT,
                    bordercolor=BORDER, insertcolor=ACCENT, padding=2)

    # The number beside a slider. Same face as a value label so it reads as a
    # readout, but it is a real entry and takes a typed value.
    style.configure("Value.TEntry", fieldbackground=PANEL_LIGHT,
                    foreground=TEXT, bordercolor=BORDER, insertcolor=ACCENT,
                    padding=1)
    style.map("Value.TEntry",
              fieldbackground=[("focus", TIMELINE_BG)],
              bordercolor=[("focus", ACCENT)])

    style.configure("TScale", background=BG, troughcolor=PANEL_LIGHT,
                    bordercolor=BORDER)
    # A slider you can actually see: an accent-coloured handle on a dark
    # trough, thick enough to grab.
    style.configure("Horizontal.TScale", background=BG,
                    troughcolor=TIMELINE_BG, bordercolor=BORDER,
                    lightcolor=ACCENT, darkcolor=ACCENT_DIM,
                    sliderthickness=16, sliderrelief="flat")

    style.configure("TScrollbar", background=PANEL_LIGHT, troughcolor=PANEL,
                    bordercolor=BORDER, arrowcolor=TEXT_DIM)
    style.map("TScrollbar", background=[("active", "#4a4a4a")])

    style.configure("TCombobox", fieldbackground=PANEL_LIGHT, background=PANEL_LIGHT,
                    foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER)
    style.map("TCombobox", fieldbackground=[("readonly", PANEL_LIGHT)],
              selectbackground=[("readonly", SELECT)])

    # Timeline scrollbar: wider than default so the thumb stays grabbable when
    # zoomed deep into an hour-long episode.
    style.configure("Fat.Horizontal.TScrollbar", background=PANEL_LIGHT,
                    troughcolor=PANEL, bordercolor=BORDER, arrowcolor=TEXT,
                    arrowsize=18, width=18)
    style.map("Fat.Horizontal.TScrollbar", background=[("active", ACCENT_DIM)])

    style.configure("TProgressbar", background=ACCENT, troughcolor=PANEL_LIGHT,
                    bordercolor=BORDER)

    style.configure("TNotebook", background=BG, bordercolor=BORDER)
    style.configure("TNotebook.Tab", background=PANEL, foreground=TEXT_DIM,
                    padding=(14, 6))
    style.map("TNotebook.Tab", background=[("selected", BG)],
              foreground=[("selected", ACCENT)])

    style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=TEXT, bordercolor=BORDER, rowheight=22)
    style.map("Treeview", background=[("selected", SELECT)])
    style.configure("Treeview.Heading", background=PANEL_LIGHT, foreground=TEXT_DIM,
                    font=FONT_SMALL)

    return style


def menu_options():
    """
    Shared options for every dropdown.

    selectcolor is the one that matters: without it the check-mark indicator is
    drawn in a light default colour with a light glyph on top, so ticks in the
    Export menu were effectively invisible against the dark pane.
    """
    return dict(
        tearoff=0,
        background=PANEL,
        foreground=TEXT,
        activebackground=SELECT,
        activeforeground="#ffffff",
        selectcolor=ACCENT,
        disabledforeground="#5a5a5a",
        borderwidth=0,
        activeborderwidth=0,
    )


def listbox_options():
    """kwargs for tk.Listbox, which is a classic widget and ignores ttk styling."""
    return {
        "background": PANEL,
        "foreground": TEXT,
        "selectbackground": SELECT,
        "selectforeground": "#ffffff",
        "highlightthickness": 0,
        "borderwidth": 0,
        "font": FONT,
    }


def text_options():
    """kwargs for tk.Text / tk.Entry."""
    return {
        "background": PANEL,
        "foreground": TEXT,
        "insertbackground": ACCENT,
        "selectbackground": SELECT,
        "highlightthickness": 0,
        "borderwidth": 0,
        "font": FONT,
    }


# Status colours, shared by the loudness meters and the update indicator. The
# meters had their own copies in app.py; these are the same values, in the one
# place a widget outside app.py can reach them.
OK_GREEN = "#3fbf6f"
WARN_YELLOW = "#e8c341"
BAD_RED = "#e0503f"


class Tooltip:
    """
    Hover text for one widget.

    tkinter has no tooltip of its own, and the update indicator is a coloured
    dot whose whole meaning is otherwise a guess. Written as a class so the
    text can be re-read on every hover: the dot's tooltip changes as the update
    check completes, and a caption captured once at construction would still be
    saying "checking..." an hour later.

    `text` may be a string or a zero-argument callable.
    """

    DELAY_MS = 450          # long enough not to fire while crossing the widget

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        self._after = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        if self.window is not None:
            return
        caption = self.text() if callable(self.text) else self.text
        if not caption:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6

        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)      # no title bar, no border
        self.window.configure(background=BORDER)
        label = tk.Label(self.window, text=caption, justify="left",
                         background=PANEL_LIGHT, foreground=TEXT,
                         font=FONT_SMALL, padx=8, pady=4)
        label.pack(padx=1, pady=1)                 # 1px BORDER shows as a frame

        # Keep it on screen when the widget is near the right or bottom edge.
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        screen_w = self.widget.winfo_screenwidth()
        screen_h = self.widget.winfo_screenheight()
        x = min(x, screen_w - width - 8)
        if y + height > screen_h - 8:
            y = self.widget.winfo_rooty() - height - 6
        self.window.wm_geometry(f"+{max(0, x)}+{max(0, y)}")

    def _hide(self, _event=None):
        self._cancel()
        if self.window is not None:
            self.window.destroy()
            self.window = None


def attach_tooltip(widget, text):
    """Convenience wrapper - returns the Tooltip so callers can keep it."""
    return Tooltip(widget, text)
