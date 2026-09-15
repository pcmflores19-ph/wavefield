"""
Per-track effects: the built-in effects, any VST3 installed on this machine,
and the chain for one speaker's track.

The built-in effects (effects.py, ported from OBS) are edited here with plain
sliders. VST3 plugins are edited through their OWN native GUI - double-click
one in the chain to open it. Rebuilding parameter controls for a VST3 from the
host side was tried and dropped: the reconstructed values did not reliably
match plugin state. Our own effects have no such problem, because we know
exactly what their parameters mean.
"""

import math
import os

import tkinter as tk
from tkinter import messagebox, ttk

import effects
import value_entry
import ui_theme
from vst_host import discover_plugins, open_editor_subprocess

# The size the dialog opens at. Named because centring has to work them out
# before the window is mapped, when winfo_width() still reports 1.
# Portrait (taller than wide): the chain list is the thing that grows long
# (four, five, six slots), not wide, so the extra room is better spent as
# height than width. Capped low enough to actually fit a common 1536x864
# laptop display with room for the taskbar/title bar left over - a taller
# window looked right in isolation but had its own button row pushed off
# the bottom of the screen, unreachable, on exactly that display.
# _ensure_buttons_visible only ever grows a window that's too small for its
# content; it does nothing for one that starts too tall for the screen.
DIALOG_WIDTH = 650
DIALOG_HEIGHT = 800

# Loudness meter column, right of the plugin/chain lists - same scale and
# ballistics as the main mixer's meters (app.py), duplicated here rather than
# imported: it's a handful of pure functions, and this dialog has no other
# dependency on app.py.
METER_WIDTH = 68
METER_FLOOR_DB = -60.0
METER_DECAY = 0.25
PEAK_HOLD_TICKS = 18
METER_GREEN_MAX_DB = -18.0
METER_YELLOW_MAX_DB = -6.0
METER_GREEN = "#00A34A"       # Safe zone: dark kelly green - kept in sync
METER_YELLOW = "#CBB000"      # Caution zone: olive-tinted mustard yellow -
METER_RED = "#D1232A"         # Clip zone: crimson - with app.py's meters
METER_SCALE_TICKS = (0, -6, -12, -24, -40)
METER_TICK_MS = 60  # matches app.py's own _tick interval


def _discard(path):
    """Removes a file if it's there, never raising - cleanup only."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _trace(message):
    """
    Into the freeze log, and never fatal.

    The import is inside the try on purpose. These calls sit in the Apply path,
    and an instrumentation import that failed - a frozen build that did not
    bundle the module, say - would take Apply down with it, or worse, raise in
    _apply_done before dialog.close() and leave the progress window up with
    nothing to dismiss it. Diagnostics must never be load-bearing.
    """
    try:
        import diagnostics
        diagnostics.trace(message)
    except Exception:
        pass


class _MeasureSlot:
    """Fake chain slot used only to render a probe params grid for sizing.

    Real slots come from project.py and carry more state (dirty flags, VST3
    handles, etc.) than _render_param_grid actually touches - this supplies
    just what that method reads.
    """

    def __init__(self, key):
        self.key = key
        self.is_native = True
        self.params = effects.defaults(key)

    def mark_dirty(self):
        pass


class FxDialog(tk.Toplevel):
    def __init__(self, parent, track_name, chain, on_change=None, log=None,
                 player=None, on_replace=None, track=None):
        super().__init__(parent)
        self.title(f"Effects - {track_name}")
        self.geometry(f"{DIALOG_WIDTH}x{DIALOG_HEIGHT}")
        self.transient(parent)

        self.chain = chain
        self.on_change = on_change or (lambda: None)
        self.log = log or (lambda msg: None)
        self.player = player        # paused around plugin loads
        # Called with a brand-new TrackChain when a preset replaces this one.
        # The dialog cannot do the swap itself: the app owns the list of chains
        # and the player owns the track, and both must be updated together.
        self.on_replace = on_replace
        self.available = discover_plugins()

        # player.Track this chain belongs to - gives the loudness meter
        # access to what the live chain is actually producing. None only for
        # a hypothetical caller that does not want the meter; app.py always
        # supplies it. Auditioning a chain edit is just pressing the app's
        # own Play button - the chain runs live on every block (player.py),
        # so there is nothing separate this dialog needs to render or loop.
        self.track = track

        self._parent_window = parent
        self._build()
        self._refresh_chain()
        # Centred on the size the dialog actually NEEDS, not on winfo_width():
        # the dialog is not mapped yet at this point, so winfo_width() still
        # reports 1 - see _final_size(), which uses winfo_reqwidth()/
        # reqheight() instead, and _centre_on's own comment for why the
        # geometry request has to be size+position in one call.
        width, height = self._final_size()
        self._centre_on(parent, width, height)

    def _final_size(self):
        """
        The size the dialog opens at, and the ONLY size it will ever need:
        nothing after this grows or shrinks the window again during normal
        use (see _reserve_params_height for why the settings panel's own
        height no longer varies by effect).

        Width stays at DIALOG_WIDTH always - ttk grid columns shrink to fit a
        narrower window rather than clipping, so 650 still looks right, it's
        just tighter. Only height is allowed to grow past DIALOG_HEIGHT, for
        the settings panel reserved in _reserve_params_height.
        """
        self.update_idletasks()
        height = max(DIALOG_HEIGHT, self.winfo_reqheight())
        return DIALOG_WIDTH, height

    def _centre_on(self, parent, width=None, height=None):
        """
        Opens over the middle of the main window rather than at the top-left
        of the screen, which is where Tk puts a Toplevel that never asks.

        `width`/`height` are passed in whenever the caller already knows the
        size, because this runs before the window is mapped and the winfo_*
        measurements are not usable until it is. Size and position go out in
        ONE geometry call - setting them separately let Tk show the window at
        the old position first, which flashed.

        A third of the way down rather than halfway: the dialog is tall, and
        centring it vertically pushed the bottom button row off a short screen.
        Same placement rule as ProgressDialog.
        """
        try:
            if width is None or height is None:
                self.update_idletasks()
                width = width or self.winfo_width()
                height = height or self.winfo_height()
            parent.update_idletasks()
            x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - height) // 3
            # Never off the top or left, or the title bar becomes undraggable.
            x, y = max(0, x), max(0, y)
            self.geometry(f"{width}x{height}+{x}+{y}")
            self.update_idletasks()
            # Where the window manager ACTUALLY put it, which differs from
            # what we asked for by the frame border. Recorded so _ensure_
            # buttons_visible can tell an untouched window from a moved one.
            self._placed_at = (self.winfo_rootx(), self.winfo_rooty())
        except Exception:
            pass        # placement is cosmetic - never block the dialog

    def destroy(self):
        if getattr(self, "_meter_job", None) is not None:
            try:
                self.after_cancel(self._meter_job)
            except Exception:
                pass
        super().destroy()

    # ---------- loudness meter ----------

    @staticmethod
    def _to_db(level):
        if level <= 1e-7:
            return METER_FLOOR_DB
        return max(METER_FLOOR_DB, 20.0 * math.log10(level))

    @staticmethod
    def _db_to_fraction(db):
        return max(0.0, min(1.0, (db - METER_FLOOR_DB) / (0.0 - METER_FLOOR_DB)))

    @staticmethod
    def _level_color(db):
        if db > METER_YELLOW_MAX_DB:
            return METER_RED
        if db > METER_GREEN_MAX_DB:
            return METER_YELLOW
        return METER_GREEN

    def _meter_tick(self):
        if self._closed_check():
            return
        peak = self.track.peak_level if self.track is not None else 0.0
        peak_db = self._to_db(peak)

        state = self._meter_state
        # Rises instantly, holds briefly, falls gradually - same ballistics
        # as the main mixer's meters (app.py._update_meter_state).
        if peak_db >= state["peak"]:
            state["peak"] = peak_db
            state["hold"] = PEAK_HOLD_TICKS
        elif state["hold"] > 0:
            state["hold"] -= 1
        else:
            state["peak"] += (peak_db - state["peak"]) * METER_DECAY

        self._draw_meter()
        self._meter_job = self.after(METER_TICK_MS, self._meter_tick)

    def _draw_meter(self):
        canvas = self.meter_canvas
        canvas.delete("all")
        width = int(canvas.winfo_width()) or METER_WIDTH
        height = int(canvas.winfo_height()) or 320

        bar_top = 8
        # Leaves room below the bar for the dB reading drawn at bar_bottom+10
        # (see the create_text call at the end of this method) - at height-8
        # that text landed 2px past the canvas's own bottom edge and never
        # rendered, on every canvas height.
        bar_bottom = height - 20
        bar_height = max(1, bar_bottom - bar_top)
        x0, x1 = 6, width - 22

        canvas.create_rectangle(x0, bar_top, x1, bar_bottom,
                                fill="#0a0a0a", outline="#3a3a3a")

        def y_for(db):
            return bar_bottom - self._db_to_fraction(db) * bar_height

        state = self._meter_state
        level_db = state["peak"]
        zones = [(METER_FLOOR_DB, METER_GREEN_MAX_DB, METER_GREEN),
                 (METER_GREEN_MAX_DB, METER_YELLOW_MAX_DB, METER_YELLOW),
                 (METER_YELLOW_MAX_DB, 0.0, METER_RED)]
        for zone_lo, zone_hi, color in zones:
            if level_db <= zone_lo:
                break
            top_db = min(level_db, zone_hi)
            y_hi, y_lo = y_for(top_db), y_for(zone_lo)
            if y_lo - y_hi >= 1:
                canvas.create_rectangle(x0 + 1, y_hi, x1 - 1, y_lo,
                                        fill=color, outline="")

        for db in (METER_GREEN_MAX_DB, METER_YELLOW_MAX_DB):
            canvas.create_line(x0, y_for(db), x1, y_for(db), fill="#555")

        for db in METER_SCALE_TICKS:
            y = y_for(db)
            canvas.create_line(x1, y, x1 + 3, y, fill="#666")
            canvas.create_text(x1 + 5, y, text=f"{db}", fill="#888",
                               anchor="w", font=("TkDefaultFont", 6))

        reading = ("-inf" if state["peak"] <= METER_FLOOR_DB
                   else f"{state['peak']:.1f}")
        canvas.create_text(width / 2, bar_bottom + 10, text=reading,
                           fill=self._level_color(state["peak"]),
                           font=("TkDefaultFont", 8, "bold"))

    # ---------- layout ----------

    @staticmethod
    def _scrolled_listbox(parent, **kwargs):
        """Listbox with vertical and horizontal scrollbars that always show."""
        wrap = ttk.Frame(parent)
        # Breathing room round the text. Safe now that the group behind it is
        # the same grey as the list - the earlier dark band came from the gap
        # showing the WINDOW colour through, not from the padding itself.
        wrap.pack(fill="both", expand=True, padx=6, pady=(2, 0))
        vsb = ttk.Scrollbar(wrap, orient="vertical")
        hsb = ttk.Scrollbar(wrap, orient="horizontal")
        options = dict(ui_theme.listbox_options())
        options.update(kwargs)
        box = tk.Listbox(wrap, exportselection=False,
                         yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                         **options)
        vsb.config(command=box.yview)
        hsb.config(command=box.xview)
        box.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        return box

    def _build(self):
        self.resizable(True, True)
        self.minsize(560, 320)

        self.rowconfigure(0, weight=1)     # plugin lists
        self.columnconfigure(0, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=0)

        # Built-in effects and VST3 plugins get a box each. Tagging one list
        # with "(VST3)" on every row was noise - the split says it once.
        left = ttk.Frame(top)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        left.rowconfigure(0, weight=0)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        builtin_frame = ttk.LabelFrame(left, text="Effects",
                                       style="Flush.TLabelframe")
        builtin_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 2))
        self.builtin_list = self._scrolled_listbox(
            builtin_frame, height=len(effects.EFFECTS))
        for _key, label, _fn, _params in effects.EFFECTS:
            self.builtin_list.insert("end", label)
        self.builtin_list.bind("<Double-Button-1>", lambda e: self._add_builtin())
        self.builtin_list.bind(
            "<<ListboxSelect>>",
            lambda e: self.vst_list.selection_clear(0, "end"))
        ttk.Button(builtin_frame, text="Add to chain  ->",
                   command=self._add_builtin).pack(padx=2, pady=(3, 2))

        vst_frame = ttk.LabelFrame(
            left, text=f"VST3 plugins ({len(self.available)} found)",
            style="Flush.TLabelframe")
        vst_frame.grid(row=1, column=0, sticky="nsew")
        self.vst_list = self._scrolled_listbox(vst_frame, height=6)
        for name, _path in self.available:
            self.vst_list.insert("end", name)
        self.vst_list.bind("<Double-Button-1>", lambda e: self._add_vst())
        self.vst_list.bind(
            "<<ListboxSelect>>",
            lambda e: self.builtin_list.selection_clear(0, "end"))
        ttk.Button(vst_frame, text="Add to chain  ->",
                   command=self._add_vst).pack(padx=2, pady=(3, 2))

        # Chain (right)
        chain_frame = ttk.LabelFrame(top, text="Chain (signal flows top to bottom)",
                                     style="Flush.TLabelframe")
        chain_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))

        self.chain_list = self._scrolled_listbox(chain_frame, height=10)
        self.chain_list.bind("<Double-Button-1>", lambda e: self._open_editor())

        chain_buttons = ttk.Frame(chain_frame)
        chain_buttons.pack(fill="x", padx=2, pady=(3, 2))
        ttk.Button(chain_buttons, text="Up", width=5,
                   command=lambda: self._move(-1)).pack(side="left")
        ttk.Button(chain_buttons, text="Down", width=6,
                   command=lambda: self._move(1)).pack(side="left", padx=2)
        ttk.Button(chain_buttons, text="Bypass", width=8,
                   command=self._toggle_bypass).pack(side="left", padx=2)
        ttk.Button(chain_buttons, text="Remove", width=8,
                   command=self._remove).pack(side="left")

        self.chain_enabled = tk.BooleanVar(value=self.chain.enabled)
        ttk.Checkbutton(chain_frame, text="Chain active on this track",
                        variable=self.chain_enabled,
                        command=self._toggle_chain).pack(anchor="w", padx=2, pady=(0, 2))

        # Loudness meter (right) - what this track is actually contributing to
        # the mix (post-VST, post-mute, post-fader; see player.Track), so it
        # reads whatever is really coming out during playback, not just the
        # raw source.
        meter_frame = ttk.LabelFrame(top, text="Level", style="Flush.TLabelframe")
        meter_frame.grid(row=0, column=2, sticky="ns", padx=(2, 0))
        self.meter_canvas = tk.Canvas(meter_frame, width=METER_WIDTH,
                                      height=320, bg="#1a1a1a", highlightthickness=0)
        self.meter_canvas.pack(fill="y", expand=True, padx=4, pady=4)
        self._meter_state = {"peak": METER_FLOOR_DB, "hold": 0}
        self._meter_job = self.after(METER_TICK_MS, self._meter_tick)

        # Sliders for whichever built-in effect is selected. Empty for a VST3,
        # which has its own window instead.
        self.params_frame = ttk.LabelFrame(self, text="Settings",
                                           style="Flush.TLabelframe")
        self.params_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(4, 0))
        self._reserve_params_height()
        self.chain_list.bind("<<ListboxSelect>>", lambda e: self._show_params())

        bottom = ttk.Frame(self)
        bottom.grid(row=2, column=0, sticky="ew", padx=6, pady=(4, 6))

        # Buttons before the caption. pack gives space in the order it is
        # asked for, so a long label packed first takes the whole row and
        # squeezes everything after it to a single pixel - which is exactly
        # what had happened to Close, invisibly, until Presets landed beside
        # it and made the bug obvious.
        ttk.Button(bottom, text="Open plugin GUI",
                   command=self._open_editor).pack(side="left")
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")
        self.presets_button = ttk.Menubutton(bottom, text="Presets  ▾",
                                             style="Menu.TMenubutton",
                                             direction="above")
        self.presets_button.pack(side="right", padx=(0, 8))
        self._refresh_presets_menu()

        # Last, and allowed to be clipped: it is a hint, not a control.
        ttk.Label(bottom,
                  text="Built-in effects use the sliders above. Double-click a "
                       "VST3 in the chain to open its own window.",
                  foreground="#888").pack(side="left", padx=10)

    # ---------- presets ----------

    def _refresh_presets_menu(self):
        """
        Rebuilt every time it changes, because tk.Menu has no tidy way to
        replace one section and the list is short.
        """
        import fx_presets

        menu = tk.Menu(self.presets_button, **ui_theme.menu_options())
        menu.add_command(label="Save this chain as a preset...",
                         command=self._save_preset)

        names = fx_presets.names()
        if names:
            menu.add_separator()
            for name in names:
                # describe() reads the file rather than loading any plugins, so
                # building this menu never touches a VST3.
                menu.add_command(
                    label=f"{name}   ({fx_presets.describe(name)})",
                    command=lambda n=name: self._load_preset(n))
            menu.add_separator()
            delete_menu = tk.Menu(menu, **ui_theme.menu_options())
            for name in names:
                delete_menu.add_command(
                    label=name, command=lambda n=name: self._delete_preset(n))
            menu.add_cascade(label="Delete", menu=delete_menu)
        else:
            menu.add_separator()
            menu.add_command(label="(no presets saved yet)", state="disabled")

        self.presets_button.configure(menu=menu)
        self._presets_menu = menu       # keep a reference so tk cannot free it

    def _save_preset(self):
        import fx_presets
        from tkinter import simpledialog

        if not self.chain.slots:
            messagebox.showinfo("Nothing to save",
                                "Add an effect to the chain first.",
                                parent=self)
            return
        name = simpledialog.askstring(
            "Save preset",
            "Name this chain, so you can use it again on the next episode:",
            parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        if name in fx_presets.names() and not messagebox.askyesno(
                "Replace preset?",
                f'"{name}" already exists. Replace it?', parent=self):
            return
        if fx_presets.save(name, self.chain):
            self.log(f'Saved preset "{name}": {self.chain.describe()}')
            self._refresh_presets_menu()
        else:
            messagebox.showerror("Could not save",
                                 "The preset file could not be written.",
                                 parent=self)

    def _load_preset(self, name):
        """
        Replaces this track's chain with the preset.

        The chain object itself is swapped, so both the app's list and the
        player's track have to be pointed at the new one - updating only one
        leaves the audio callback mixing through the old chain, which sounds
        like the preset silently did nothing.
        """
        import fx_presets

        if self.chain.slots and not messagebox.askyesno(
                "Replace the current chain?",
                f'Loading "{name}" replaces the effects on this track.',
                parent=self):
            return

        # Loading a VST3 while the audio stream is running is a native crash,
        # so stop first - same reason _add_vst does.
        if self.player is not None:
            try:
                self.player.stop()
            except Exception:
                pass

        chain = fx_presets.load(name, log=self.log)
        if chain is None:
            messagebox.showerror("Preset missing",
                                 f'"{name}" could not be read.', parent=self)
            return

        # Keep this track's own on/off state; the preset is a recipe, not a
        # decision about whether effects are running.
        chain.enabled = self.chain.enabled
        if self.on_replace is not None:
            self.on_replace(chain)
        self.chain = chain
        self.log(f'Loaded preset "{name}": {chain.describe()}')
        self._refresh_chain()

    def _delete_preset(self, name):
        import fx_presets

        if not messagebox.askyesno("Delete preset?",
                                   f'Delete "{name}"? This cannot be undone.',
                                   parent=self):
            return
        fx_presets.delete(name)
        self.log(f'Deleted preset "{name}"')
        self._refresh_presets_menu()

    # ---------- chain operations ----------

    def _show_params(self):
        """Rebuilds the settings panel for whatever is selected in the chain."""
        for child in self.params_frame.winfo_children():
            child.destroy()

        index = self._selected_index()
        slot = self.chain.slots[index] if index is not None and \
            index < len(self.chain.slots) else None
        if slot is None or not getattr(slot, "is_native", False):
            ttk.Label(self.params_frame,
                      text="Select an effect to change its settings. "
                           "VST3 plugins open their own window.",
                      style="PanelDim.TLabel").pack(anchor="w", padx=8, pady=4)
            self._ensure_buttons_visible()
            return

        self._render_param_grid(slot)
        self._ensure_buttons_visible()

    def _render_param_grid(self, slot):
        """Builds the slider grid for `slot` into params_frame."""
        _label, _fn, spec = effects.BY_KEY[slot.key]
        grid = ttk.Frame(self.params_frame, style="Panel.TFrame")
        grid.pack(fill="x", padx=8, pady=4)
        grid.columnconfigure(1, weight=1)

        for row, (name, caption, lo, hi, default, unit) in enumerate(spec):
            ttk.Label(grid, text=caption, width=16, style="Panel.TLabel").grid(
                row=row, column=0, sticky="w", pady=2, padx=(0, 4))

            var = tk.DoubleVar(value=float(slot.params.get(name, default)))

            def commit(value, s=slot, n=name, lo=lo, hi=hi):
                s.params[n] = max(lo, min(hi, float(value)))
                s.mark_dirty()
                self.on_change()

            scale = ttk.Scale(grid, from_=lo, to=hi, orient="horizontal",
                              variable=var,
                              command=lambda _v, v=var, c=commit: c(v.get()))
            scale.grid(row=row, column=1, sticky="ew", padx=8)

            def reset_to_default(_event=None, v=var, d=default, c=commit):
                v.set(d)
                c(d)

            scale.bind("<Double-Button-1>", reset_to_default)

            # A slider is fine for a rough sweep and hopeless for "-18".
            entry = value_entry.attach(grid, var, lo, hi, on_commit=commit,
                                       width=7, fmt=lambda v: f"{v:.2f}")
            entry.grid(row=row, column=2, sticky="e")

            ttk.Label(grid, text=unit, width=5,
                      style="PanelDim.TLabel").grid(row=row, column=3,
                                                    sticky="w", padx=(4, 4))

        ttk.Button(grid, text="Reset to defaults", width=18,
                   command=lambda s=slot: self._reset_params(s)).grid(
                       row=len(spec), column=1, sticky="w", padx=8, pady=(8, 2))

    def _reserve_params_height(self):
        """
        Locks the settings panel to the height of its largest possible
        content - the built-in effect with the most sliders - instead of
        letting it grow with whatever is selected.

        Effects range from 1 slider (Gain) to 5 (Noise Gate, Compressor,
        Expander). Without this, params_frame's required height changed on
        every click between a short effect and a tall one, and
        _ensure_buttons_visible read that as "the window is now too small"
        and grew + recentred the whole dialog - so just browsing the chain
        made the window visibly jump size.
        """
        max_key = max(effects.BY_KEY, key=lambda k: len(effects.BY_KEY[k][2]))
        probe = _MeasureSlot(max_key)
        self._render_param_grid(probe)
        self.update_idletasks()
        height = self.params_frame.winfo_reqheight()
        for child in self.params_frame.winfo_children():
            child.destroy()
        self.params_frame.grid_propagate(False)
        self.params_frame.configure(height=height)

    def _ensure_buttons_visible(self):
        """
        Grows the window's HEIGHT ONLY if the settings panel just got tall
        enough to push the Close/Presets row below the visible area. Width
        is never grown here: the list/meter columns' natural reqwidth is
        wider than DIALOG_WIDTH regardless of anything selected (ttk grid
        columns shrink to fit rather than clipping), so checking width here
        used to grow the dialog to that wider natural size on the first
        click - every time, unconditionally - which read as the window
        randomly resizing when an effect was selected. Never shrinks a
        height the user already chose.
        """
        self.update_idletasks()
        needed_h = self.winfo_reqheight()
        current_w = self.winfo_width()
        current_h = self.winfo_height()
        if needed_h > current_h:
            grown_w = current_w
            grown_h = needed_h
            parent = getattr(self, "_parent_window", None)
            placed = getattr(self, "_placed_at", None)
            here = (self.winfo_rootx(), self.winfo_rooty())
            if parent is not None and placed == here:
                # Untouched since we placed it, so keep it centred at the new
                # size instead of letting it grow down and to the right.
                self._centre_on(parent, grown_w, grown_h)
            else:
                # The user has moved it. Respect that and only resize.
                self.geometry(f"{grown_w}x{grown_h}")

    def _reset_params(self, slot):
        slot.params = dict(effects.defaults(slot.key))
        slot.mark_dirty()
        self.on_change()
        self._show_params()

    def _selected_index(self):
        sel = self.chain_list.curselection()
        return sel[0] if sel else None

    def _refresh_chain(self):
        keep = self._selected_index()
        self.chain_list.delete(0, "end")
        for slot in self.chain.slots:
            label = f"{slot.name}   [bypassed]" if slot.bypassed else slot.name
            self.chain_list.insert("end", label)
        if keep is not None and keep < self.chain_list.size():
            self.chain_list.selection_set(keep)
        self.on_change()
        if hasattr(self, "params_frame"):
            self._show_params()

    def _add_builtin(self):
        sel = self.builtin_list.curselection()
        if not sel:
            return
        key = effects.EFFECTS[sel[0]][0]
        self.chain.add_native(key)
        self.log(f"Added {effects.BY_KEY[key][0]}.")
        self._refresh_chain()
        # Select what was just added, so its sliders appear straight away.
        self.chain_list.selection_clear(0, "end")
        self.chain_list.selection_set(len(self.chain.slots) - 1)
        self._show_params()

    def _add_vst(self):
        sel = self.vst_list.curselection()
        if not sel:
            return
        name, path = self.available[sel[0]]
        # Loading a plugin while the audio callback is inside one is a native
        # crash, so playback stops first.
        if self.player and self.player.is_playing:
            self.player.stop()
        try:
            self.chain.add(name, path)
            self.log(f"Added {name}.")
        except Exception as exc:
            messagebox.showerror("Could not load plugin",
                                 f"{name}\n\n{exc}", parent=self)
        finally:
            self._refresh_chain()

    def _remove(self):
        i = self._selected_index()
        if i is None:
            return
        self.log(f"Removed {self.chain.slots[i].name} from the chain.")
        self.chain.remove(i)
        self._refresh_chain()

    def _move(self, delta):
        i = self._selected_index()
        if i is None:
            return
        new_index = self.chain.move(i, delta)
        self._refresh_chain()
        self.chain_list.selection_clear(0, "end")
        self.chain_list.selection_set(new_index)

    def _toggle_bypass(self):
        i = self._selected_index()
        if i is None:
            return
        slot = self.chain.slots[i]
        slot.bypassed = not slot.bypassed
        slot.mark_dirty()
        self._refresh_chain()

    def _toggle_chain(self):
        self.chain.enabled = self.chain_enabled.get()
        self.on_change()

    def _closed_check(self):
        try:
            return not self.winfo_exists()
        except Exception:
            return True

    # ---------- native editor ----------

    def _open_editor(self):
        i = self._selected_index()
        if i is None:
            return
        slot = self.chain.slots[i]
        if getattr(slot, "is_native", False):
            self._show_params()
            return

        def done():
            # Fires on open_editor_subprocess's worker thread, not the Tk
            # thread - touching widgets here directly is a silent race that
            # sometimes just drops the waveform/FX-button refresh.
            self.after(0, lambda: (
                self.log(f"{slot.name}: settings applied."),
                self.on_change(),
            ))

        def failed(message):
            self.after(0, lambda: self.log(
                f"{slot.name}: could not open plugin GUI - {message}"))

        self.log(f"Opening {slot.name} GUI in a separate window "
                 "(close it to apply your changes).")
        open_editor_subprocess(slot, on_done=done, on_error=failed)
