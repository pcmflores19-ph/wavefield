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

import tkinter as tk
from tkinter import messagebox, ttk

import effects
import value_entry
import ui_theme
from vst_host import discover_plugins, open_editor_subprocess


class FxDialog(tk.Toplevel):
    def __init__(self, parent, track_name, chain, on_change=None, log=None,
                 player=None, on_replace=None):
        super().__init__(parent)
        self.title(f"Effects - {track_name}")
        self.geometry("720x420")
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

        self._build()
        self._refresh_chain()

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

        # Sliders for whichever built-in effect is selected. Empty for a VST3,
        # which has its own window instead.
        self.params_frame = ttk.LabelFrame(self, text="Settings",
                                           style="Flush.TLabelframe")
        self.params_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(4, 0))
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
                      style="PanelDim.TLabel").pack(anchor="w", padx=2, pady=4)
            return

        _label, _fn, spec = effects.BY_KEY[slot.key]
        grid = ttk.Frame(self.params_frame, style="Panel.TFrame")
        grid.pack(fill="x", padx=2, pady=4)
        grid.columnconfigure(1, weight=1)

        for row, (name, caption, lo, hi, default, unit) in enumerate(spec):
            ttk.Label(grid, text=caption, width=16, style="Panel.TLabel").grid(
                row=row, column=0, sticky="w", pady=2)

            var = tk.DoubleVar(value=float(slot.params.get(name, default)))

            def commit(value, s=slot, n=name, lo=lo, hi=hi):
                s.params[n] = max(lo, min(hi, float(value)))
                self.on_change()

            scale = ttk.Scale(grid, from_=lo, to=hi, orient="horizontal",
                              variable=var,
                              command=lambda _v, v=var, c=commit: c(v.get()))
            scale.grid(row=row, column=1, sticky="ew", padx=8)

            # A slider is fine for a rough sweep and hopeless for "-18".
            entry = value_entry.attach(grid, var, lo, hi, on_commit=commit,
                                       width=7)
            entry.grid(row=row, column=2, sticky="e")

            ttk.Label(grid, text=unit, width=5,
                      style="PanelDim.TLabel").grid(row=row, column=3,
                                                    sticky="w", padx=(4, 0))

        ttk.Button(grid, text="Reset to defaults", width=18,
                   command=lambda s=slot: self._reset_params(s)).grid(
                       row=len(spec), column=1, sticky="w", padx=8, pady=(8, 2))

    def _reset_params(self, slot):
        slot.params = dict(effects.defaults(slot.key))
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
        self._refresh_chain()

    def _toggle_chain(self):
        self.chain.enabled = self.chain_enabled.get()
        self.on_change()

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
            self.log(f"{slot.name}: settings applied.")
            self.on_change()

        def failed(message):
            self.log(f"{slot.name}: could not open plugin GUI - {message}")

        self.log(f"Opening {slot.name} GUI in a separate window "
                 "(close it to apply your changes).")
        open_editor_subprocess(slot, on_done=done, on_error=failed)
