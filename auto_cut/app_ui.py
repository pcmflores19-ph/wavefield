"""
UI construction for the editor, kept apart from the editing logic in app.py.

Laid out like DaVinci Resolve, because that is where the exported timeline
ends up: transcript and inspector above, the timeline filling the width
beneath them. There is only one page - exporting is a menu, not somewhere you
navigate to.
"""

import os
import tkinter as tk
from tkinter import messagebox, ttk

import effects
import help_text
import links
import ui_theme
import value_entry
import version
from whisperx_runner import DEFAULT_LANGUAGE, DEFAULT_MODEL

LANE_HEIGHT = 74
RULER_HEIGHT = 18
METER_WIDTH = 68
HSCROLL_HEIGHT = 18        # fixed height for the timeline scrollbar
INSPECTOR_WIDTH = 340


class UIBuilderMixin:
    """Builds every widget. Expects the host class to provide the callbacks."""

    # ------------------------------------------------------------------ shell

    def _build_ui(self):
        ui_theme.apply(self.root)
        self.root.configure(background=ui_theme.BG)

        self._build_menu()

        # One page. Editing is the whole app; exporting is a menu, not a place
        # you navigate to.
        self.page_container = ttk.Frame(self.root)
        self.page_container.pack(fill="both", expand=True)
        self._build_edit_page(self.page_container)

        self._build_status_bar()

    def _build_menu(self):
        """
        Our own menu bar, not the operating system's.

        Windows draws the real menu bar itself and ignores tk's colours, so the
        labels sat in system colours above a dark window - and it gave no way
        to push Support to the right. (The native MFT_RIGHTJUSTIFY flag was
        tried: the call succeeds, the flag reads back set, and Windows then
        stops drawing the item at all.) A strip of Menubuttons we own solves
        the colours, the hover states and the alignment together.
        """
        bar = ttk.Frame(self.root, style="Menubar.TFrame")
        bar.pack(side="top", fill="x")
        self.menubar_frame = bar

        file_menu = tk.Menu(bar, **ui_theme.menu_options())
        file_menu.add_command(label="New project", command=self.new_project,
                              accelerator="Ctrl+N")
        file_menu.add_command(label="Open project...", command=self.open_project,
                              accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save project", command=self.save_project,
                              accelerator="Ctrl+S")
        file_menu.add_command(label="Save project as...",
                              command=self.save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_close)
        self._add_menu(bar, "File", file_menu)

        self._build_vodcast_menu(bar)
        self._build_export_menu(bar)
        self._build_help_menu(bar)
        # Packed to the right, away from the menus you actually work with.
        # Support goes on first so it stays hard right: with side="right", tk
        # places each new widget further left, so whatever is packed last sits
        # nearest the middle.
        self._build_support_menu(bar, side="right")
        self._build_update_dot(bar)

        self.root.bind("<Control-s>", lambda e: self.save_project())
        self.root.bind("<Control-o>", lambda e: self.open_project())
        self.root.bind("<Control-n>", lambda e: self.new_project())

    def _build_update_dot(self, bar):
        """
        A coloured dot that says whether this copy is the newest one.

        Wavefield is installed by people who will never think to look in a menu
        for a new version, and an out-of-date copy is how a fixed bug goes on
        being reported. A dot is small enough to ignore and impossible to miss
        once it turns red.

        Grey until the check finishes, so it never claims to know something it
        does not.
        """
        size = 14
        self.update_dot = tk.Canvas(bar, width=size, height=size,
                                    background=ui_theme.PANEL,
                                    highlightthickness=0, borderwidth=0)
        self._update_dot_item = self.update_dot.create_oval(
            3, 3, size - 3, size - 3,
            fill=ui_theme.TEXT_DIM, outline="")
        self.update_dot.pack(side="right", padx=(6, 10))

        self._update_status = "unknown"
        self._update_release = None
        self._update_tip = ui_theme.attach_tooltip(
            self.update_dot, self._update_dot_caption)
        self.update_dot.bind("<Button-1>", self._on_update_dot_click)

    def _update_dot_caption(self):
        if self._update_status == "current":
            return f"{version.APP_NAME} {version.__version__} is up to date"
        if self._update_status == "available":
            newer = (self._update_release or {}).get("version", "?")
            return (f"Version {newer} is available - "
                    f"you have {version.__version__}." + chr(10) +
                    "Click for options.")
        if self._update_status == "checking":
            return "Checking for updates..."
        return ("Could not check for updates." + chr(10) +
                "Use Help > Check for updates to try again.")

    def _set_update_dot(self, status, release=None):
        """Repaints the dot. Safe to call before the widget exists."""
        if not hasattr(self, "update_dot"):
            return
        self._update_status = status
        self._update_release = release
        colour = {
            "current": ui_theme.OK_GREEN,
            "available": ui_theme.BAD_RED,
            "checking": ui_theme.TEXT_DIM,
        }.get(status, ui_theme.TEXT_DIM)
        self.update_dot.itemconfigure(self._update_dot_item, fill=colour)
        # Only worth clicking when there is something to do about it.
        self.update_dot.configure(
            cursor="hand2" if status == "available" else "")

    def _on_update_dot_click(self, event):
        if self._update_status != "available" or not self._update_release:
            return
        menu = tk.Menu(self.root, **ui_theme.menu_options())
        menu.add_command(label="Install update...",
                         command=lambda: self.install_update(self._update_release))
        menu.add_command(label="What's new",
                         command=lambda: self.show_release_notes(self._update_release))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _add_menu(self, bar, label, menu, side="left"):
        """Puts one dropdown on the bar and returns its button."""
        button = ttk.Menubutton(bar, text=label, menu=menu,
                                style="Menu.TMenubutton", direction="below")
        button.pack(side=side)
        return button

    def open_settings(self):
        from settings_dialog import SettingsDialog
        SettingsDialog(self.root, log=self.log)

    def _build_help_menu(self, menubar):
        """
        Help matters more than usual here: an installed copy has no README
        beside it, so for anyone who did not clone the repository this menu is
        the only documentation there is.
        """
        menu = tk.Menu(menubar, **ui_theme.menu_options())
        menu.add_command(label="Quick start",
                         command=lambda: self._show_help(
                             "Quick start", help_text.QUICK_START))
        menu.add_command(label="Keyboard shortcuts", accelerator="F1",
                         command=lambda: self._show_help(
                             "Keyboard shortcuts", help_text.SHORTCUTS))
        menu.add_command(label="Troubleshooting",
                         command=lambda: self._show_help(
                             "Troubleshooting", help_text.TROUBLESHOOTING))
        menu.add_command(label="Report a problem...",
                         command=self.report_problem)
        menu.add_separator()
        menu.add_command(label="Check for updates",
                         command=self.check_for_updates)
        menu.add_command(label="Project page",
                         command=self._open_project_page)
        menu.add_separator()
        menu.add_command(label=f"About {version.APP_NAME}",
                         command=lambda: self._show_help(
                             f"About {version.APP_NAME}", help_text.ABOUT))
        self._add_menu(menubar, "Help", menu)

        self.root.bind("<F1>", lambda e: self._show_help(
            "Keyboard shortcuts", help_text.SHORTCUTS))

    def report_problem(self):
        """Help > Report a problem - straight to the form, nothing saved locally."""
        self._open_url(links.REPORT_FORM)

    def _show_help(self, title, body):
        """A read-only, scrollable, resizable text window."""
        window = tk.Toplevel(self.root)
        window.title(f"{version.APP_NAME} - {title}")
        window.configure(background=ui_theme.BG)
        window.geometry("760x620")
        window.transient(self.root)

        frame = ttk.Frame(window, style="Panel.TFrame")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        scroll = ttk.Scrollbar(frame, orient="vertical")
        scroll.pack(side="right", fill="y")
        text = tk.Text(frame, wrap="word", yscrollcommand=scroll.set,
                       padx=12, pady=10, **ui_theme.text_options())
        text.pack(side="left", fill="both", expand=True)
        scroll.config(command=text.yview)

        text.insert("1.0", body)
        # Read-only, but still selectable and copyable - disabling the widget
        # outright would stop people copying an error message out of it.
        text.configure(state="disabled")

        ttk.Button(window, text="Close", width=12,
                   command=window.destroy).pack(pady=(0, 10))
        window.bind("<Escape>", lambda e: window.destroy())
        text.focus_set()

    def _open_url(self, url):
        import webbrowser
        webbrowser.open(url)

    def _open_project_page(self):
        self._open_url(version.PROJECT_URL)

    def _open_support_link(self, title, url):
        """
        Opens a Support link, or explains itself if it was never filled in.

        Sending someone to example.com is worse than saying nothing, so an
        unedited placeholder says what it is instead.
        """
        if links.is_placeholder(url):
            messagebox.showinfo(
                title,
                f"This link has not been set up yet." + chr(10) * 2 +
                f"Whoever built this copy needs to put the real {title} "
                f"address into auto_cut/links.py.")
            return
        self._open_url(url)

    def _build_support_menu(self, menubar, side="left"):
        """Where people can find, and support, the podcast behind the app."""
        menu = tk.Menu(menubar, **ui_theme.menu_options())
        for label, url in links.SUPPORT_MENU:
            if label is None:
                menu.add_separator()
                continue
            menu.add_command(
                label=label,
                command=lambda t=label, u=url: self._open_support_link(t, u))
        # Added last, so it sits to the right of the working menus. Windows
        # will not push it flush against the right edge: MFT_RIGHTJUSTIFY is a
        # legacy flag that themed menu bars no longer honour - setting it
        # succeeds, and the item then stops being drawn at all.
        self._add_menu(menubar, f"Support {links.PODCAST_NAME}", menu, side)
    def _build_status_bar(self):
        bar = ttk.Frame(self.root, style="Panel.TFrame")
        bar.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(bar, text="No project", style="PanelDim.TLabel")
        self.status_label.pack(side="left", padx=10, pady=3)
        # Progress and an elapsed clock sit on the right, visible from every
        # page - a long job must never look like nothing is happening.
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=200)
        self.progress.pack(side="right", padx=10, pady=3)
        self.busy_label = ttk.Label(bar, text="", style="Value.TLabel",
                                    background=ui_theme.PANEL)
        self.busy_label.pack(side="right", padx=4)

    # -------------------------------------------------------------- edit page

    def _build_edit_page(self, parent):
        # A paned window so the timeline can be dragged as large as needed -
        # previously the waveform was squeezed to 92px with no way to grow it.
        # The timeline's scrollbar and zoom controls live OUTSIDE the paned
        # window, pinned to the bottom of the page. Inside it they competed for
        # space with the transcript and camera strip, and dragging the sash down
        # clipped them away entirely. ttk.PanedWindow panes have no minsize, so
        # keeping them out of the pane is the only way to guarantee they stay.
        self.timeline_footer = ttk.Frame(parent, style="Panel.TFrame")
        self.timeline_footer.pack(side="bottom", fill="x", padx=6, pady=(0, 6))

        pane = ttk.PanedWindow(parent, orient="vertical")
        pane.pack(fill="both", expand=True)
        self.edit_pane = pane

        upper = ttk.Frame(pane)
        pane.add(upper, weight=3)
        lower = ttk.Frame(pane)
        pane.add(lower, weight=2)

        # Horizontal sash between the inspector and the transcript/log area,
        # so a long filename or a wide mixer strip isn't stuck at whatever
        # width happened to be chosen up front.
        upper_pane = ttk.PanedWindow(upper, orient="horizontal")
        upper_pane.pack(fill="both", expand=True)

        # ttk.PanedWindow panes have no minsize (same limitation noted above
        # for the vertical pane) - nothing stops the sash being dragged to
        # an unusably narrow width, same as the existing vertical split.
        left = ttk.Frame(upper_pane)
        upper_pane.add(left, weight=3)

        self._build_inspector(upper_pane)

        # Transcript, beside the audio rather than on another page.
        transcript = ttk.Frame(left, style="Panel.TFrame")
        transcript.pack(fill="both", expand=True, padx=(6, 3), pady=(0, 6))
        header = ttk.Frame(transcript, style="Panel.TFrame")
        header.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(header, text="TRANSCRIPT", style="PanelDim.TLabel").pack(side="left")
        ttk.Label(header, style="PanelDim.TLabel",
                  text="   follows playback - double-click a line to seek"
                  ).pack(side="left")
        ttk.Button(header, text="Save text changes", width=17,
                   command=self.apply_transcript_edits).pack(side="right")
        ttk.Label(header, style="PanelDim.TLabel",
                  text="edits here are only kept once saved  ").pack(side="right")

        body = ttk.Frame(transcript, style="Panel.TFrame")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        scroll = ttk.Scrollbar(body, orient="vertical")
        scroll.pack(side="right", fill="y")
        self.transcript_text = tk.Text(body, wrap="word", undo=True, height=6,
                                       yscrollcommand=scroll.set,
                                       **ui_theme.text_options())
        self.transcript_text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.transcript_text.yview)
        self.transcript_text.tag_configure("stamp", foreground=ui_theme.ACCENT)
        self.transcript_text.tag_configure("current", background=ui_theme.SELECT,
                                           foreground="#ffffff")
        self.transcript_text.bind("<Double-Button-1>", self._on_transcript_click)
        # A Text widget's tags belong to characters, not regions: deleting a
        # character at the edge of the "stamp" tag shrinks it, and text typed
        # back into that gap does not reacquire it - it just comes out in the
        # default colour. Reapplying the tag after every edit is simpler than
        # trying to out-think Tk's tag gravity.
        self.transcript_text.bind("<KeyRelease>",
                                  lambda _e: self._retag_transcript_stamps())
        self.transcript_text.bind("<<Paste>>",
                                  lambda _e: self.root.after_idle(self._retag_transcript_stamps))

        # The log used to live on the export page. With that page gone it has
        # to be here: every long job - analysis, transcription, rendering -
        # reports through it, and without it the app looks frozen while it
        # works. Fixed height, so it never steals room from the transcript.
        log_frame = ttk.Frame(left, style="Panel.TFrame")
        log_frame.pack(fill="x", padx=(6, 3), pady=(0, 6))
        ttk.Label(log_frame, text="LOG", style="PanelDim.TLabel").pack(
            anchor="w", padx=8, pady=(6, 2))
        log_body = ttk.Frame(log_frame, style="Panel.TFrame")
        log_body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        log_scroll = ttk.Scrollbar(log_body, orient="vertical")
        log_scroll.pack(side="right", fill="y")
        self.log_text = tk.Text(log_body, height=6, wrap="word",
                                yscrollcommand=log_scroll.set,
                                **ui_theme.text_options())
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)

        self._build_timeline(lower)

    def _scrollable(self, parent, width=None):
        """
        A vertically scrolling panel. The inspector outgrew its fixed height and
        silently clipped the mixer - FX buttons included - so anything that can
        overflow lives in one of these now.
        """
        outer = ttk.Frame(parent, style="Panel.TFrame")
        if width:
            outer.configure(width=width)
            outer.pack_propagate(False)

        canvas = tk.Canvas(outer, background=ui_theme.PANEL, highlightthickness=0)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas, style="Panel.TFrame")
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(window, width=e.width))
        # Tagged so the window-wide wheel handler can find the right scroller
        # from whatever widget the pointer happens to be over.
        for widget in (outer, canvas, inner):
            widget._wheel_scrolls = canvas
        return outer, inner

    def _build_inspector(self, parent):
        """
        `parent` is the horizontal ttk.PanedWindow built in _build_edit_page,
        not a plain container - added as a pane (not packed) so its width is
        the draggable sash position, not a fixed constant. INSPECTOR_WIDTH is
        still the width it starts at.
        """
        outer, inspector = self._scrollable(parent, width=INSPECTOR_WIDTH)
        parent.add(outer, weight=1)

        # --- media
        ttk.Label(inspector, text="MEDIA", style="PanelDim.TLabel").pack(
            anchor="w", padx=8, pady=(8, 2))
        row = ttk.Frame(inspector, style="Panel.TFrame")
        row.pack(fill="x", padx=8)
        self.files_list = tk.Listbox(row, height=3, **ui_theme.listbox_options())
        self.files_list.pack(side="left", fill="both", expand=True)
        buttons = ttk.Frame(row, style="Panel.TFrame")
        buttons.pack(side="right", padx=(4, 0))
        ttk.Button(buttons, text="Add...", width=8,
                   command=self.add_files).pack(fill="x", pady=1)
        ttk.Button(buttons, text="Remove", width=8,
                   command=self.remove_selected).pack(fill="x", pady=1)
        ttk.Button(buttons, text="Up", width=8,
                   command=self.move_up).pack(fill="x", pady=1)

        # Analysis is no longer a button. Waveforms and speech detection are
        # what every other feature is built on, so making the user ask for them
        # only ever produced an app that looked broken until you found the
        # button. Adding a recording starts it.
        #
        # What is left are the things that are genuinely a choice: transcribing
        # (minutes of GPU time, and not everyone wants a transcript) and the
        # two automatic edits.
        actions = ttk.Frame(inspector, style="Panel.TFrame")
        actions.pack(fill="x", padx=8, pady=(6, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self.transcribe_button = ttk.Button(
            actions, text="Transcribe",
            command=self.open_transcribe_dialog)
        self.transcribe_button.grid(row=0, column=0, sticky="ew", padx=1, pady=1)

        # Language and model are chosen in the Transcribe dialog now - two
        # places to set one thing, one of which did nothing until you pressed a
        # button elsewhere, was worse than one. The variables stay because
        # projects save and restore them.
        self.language = tk.StringVar(value=DEFAULT_LANGUAGE)
        self.whisper_model = tk.StringVar(value=DEFAULT_MODEL)

        self.sync_button = ttk.Button(
            actions, text="Sync",
            command=self.open_sync_dialog)
        self.sync_button.grid(row=0, column=1, sticky="ew", padx=1, pady=1)

        self.auto_cut_on = tk.BooleanVar(value=False)
        self.auto_cut_button = ttk.Checkbutton(
            actions, text="Auto-cut", style="Toggle.TCheckbutton",
            variable=self.auto_cut_on, command=self._on_auto_cut_toggle)
        self.auto_cut_button.grid(row=1, column=0, sticky="ew", padx=1, pady=1)

        self.auto_mute_on = tk.BooleanVar(value=False)
        self.auto_mute_button = ttk.Checkbutton(
            actions, text="Auto-mute", style="Toggle.TCheckbutton",
            variable=self.auto_mute_on, command=self._on_auto_mute_toggle)
        self.auto_mute_button.grid(row=1, column=1, sticky="ew", padx=1, pady=1)

        ttk.Separator(inspector).pack(fill="x", padx=8, pady=8)

        # --- cutting
        ttk.Label(inspector, text="DEAD AIR", style="PanelDim.TLabel").pack(
            anchor="w", padx=8)
        self.aggressiveness = tk.IntVar(value=50)
        aggr_row = ttk.Frame(inspector, style="Panel.TFrame")
        aggr_row.pack(fill="x", padx=8, pady=(2, 0))
        self.slider = ttk.Scale(aggr_row, from_=0, to=100, orient="horizontal",
                                variable=self.aggressiveness,
                                command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True)
        value_entry.attach(aggr_row, self.aggressiveness, 0, 100, width=4,
                           on_commit=lambda _v: self._on_slider(None)
                           ).pack(side="left", padx=(6, 0))
        self.aggr_label = ttk.Label(inspector, text="", style="Value.TLabel",
                                    background=ui_theme.PANEL)
        self.aggr_label.pack(anchor="w", padx=8)
        self.aggr_detail = ttk.Label(inspector, text="", style="PanelDim.TLabel",
                                     wraplength=INSPECTOR_WIDTH - 40,
                                     justify="left")
        self.aggr_detail.pack(anchor="w", padx=8)
        self.summary_label = ttk.Label(inspector, text="Add recordings to see the cuts.",
                                       style="PanelDim.TLabel",
                                       wraplength=INSPECTOR_WIDTH - 40,
                                       justify="left")
        self.summary_label.pack(anchor="w", padx=8, pady=(2, 6))

        ttk.Separator(inspector).pack(fill="x", padx=8, pady=8)

        # --- selection tools
        ttk.Label(inspector, text="SELECTION", style="PanelDim.TLabel").pack(
            anchor="w", padx=8)
        self.selection_label = ttk.Label(inspector, text="Drag across a lane.",
                                         style="PanelDim.TLabel",
                                         wraplength=INSPECTOR_WIDTH - 40,
                                         justify="left")
        self.selection_label.pack(anchor="w", padx=8, pady=(0, 4))

        grid = ttk.Frame(inspector, style="Panel.TFrame")
        grid.pack(fill="x", padx=8)
        self.delete_button = ttk.Button(grid, text="Delete (q)", width=15,
                                        state="disabled",
                                        command=self.delete_selection)
        self.delete_button.grid(row=0, column=0, padx=1, pady=1)
        self.restore_button = ttk.Button(grid, text="Restore (w)", width=15,
                                         state="disabled",
                                         command=self.restore_selection)
        self.restore_button.grid(row=0, column=1, padx=1, pady=1)
        self.mute_button = ttk.Button(grid, text="Mute lane (a)", width=15,
                                      state="disabled",
                                      command=self.mute_selection)
        self.mute_button.grid(row=1, column=0, padx=1, pady=1)
        self.unmute_button = ttk.Button(grid, text="Unmute lane (s)", width=15,
                                        state="disabled",
                                        command=self.unmute_selection)
        self.unmute_button.grid(row=1, column=1, padx=1, pady=1)

        undo_row = ttk.Frame(inspector, style="Panel.TFrame")
        undo_row.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Button(undo_row, text="Undo (z)", width=15,
                   command=self.undo_edit).pack(side="left", padx=1)
        ttk.Button(undo_row, text="Clear all (x)", width=15,
                   command=self.clear_edits).pack(side="left", padx=1)
        self.edits_label = ttk.Label(inspector, text="", style="PanelDim.TLabel")
        self.edits_label.pack(anchor="w", padx=8, pady=(4, 0))

        ttk.Separator(inspector).pack(fill="x", padx=8, pady=8)

        # --- mixer
        ttk.Label(inspector, text="MIXER   (monitoring only)",
                  style="PanelDim.TLabel").pack(anchor="w", padx=8)
        self.mixer_frame = ttk.Frame(inspector, style="Panel.TFrame")
        self.mixer_frame.pack(fill="x", padx=8, pady=(4, 12))

    def _build_timeline(self, parent):
        timeline = ttk.Frame(parent, style="Panel.TFrame")
        timeline.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Transport
        transport = ttk.Frame(timeline, style="Panel.TFrame")
        transport.pack(fill="x", padx=6, pady=4)
        ttk.Button(transport, text="|<", width=4,
                   command=self._go_start).pack(side="left")
        ttk.Button(transport, text="<<10s", width=7,
                   command=lambda: self._skip(-10)).pack(side="left", padx=2)
        self.play_button = ttk.Button(transport, text="Play (space)", width=13,
                                      style="Accent.TButton",
                                      command=self.toggle_play)
        self.play_button.pack(side="left", padx=2)
        ttk.Button(transport, text="10s>>", width=7,
                   command=lambda: self._skip(10)).pack(side="left", padx=2)
        ttk.Button(transport, text=">|", width=4,
                   command=self._go_end).pack(side="left")
        ttk.Button(transport, text="Stop", width=6,
                   command=self.stop_audio).pack(side="left", padx=(2, 10))
        self.time_label = ttk.Label(transport, text="0:00 / 0:00",
                                    style="Value.TLabel",
                                    background=ui_theme.PANEL)
        self.time_label.pack(side="left")

        self.edited_mode = tk.BooleanVar(value=True)
        mode = ttk.Frame(transport, style="Panel.TFrame")
        mode.pack(side="right")
        ttk.Radiobutton(mode, text="Edited", variable=self.edited_mode, value=True,
                        command=self._on_mode_change).pack(side="right")
        ttk.Radiobutton(mode, text="Raw", variable=self.edited_mode, value=False,
                        command=self._on_mode_change).pack(side="right", padx=4)
        ttk.Label(mode, text="Monitor:", style="PanelDim.TLabel").pack(side="right",
                                                                      padx=6)

        # Packing order matters here. Tk squeezes whatever was packed LAST when
        # the parent runs out of room, so the fixed-height rows (scrollbar, zoom)
        # are claimed from the bottom FIRST and the waveform is packed last with
        # expand=True. Dragging the pane sash then resizes the waveform and
        # leaves the scrollbar alone - previously it crushed the scrollbar.
        footer = self.timeline_footer
        zoom = ttk.Frame(footer, style="Panel.TFrame")
        zoom.pack(side="bottom", fill="x", padx=6, pady=4)

        hscroll_row = tk.Frame(footer, height=HSCROLL_HEIGHT,
                               background=ui_theme.PANEL)
        hscroll_row.pack(side="bottom", fill="x", padx=6, pady=(4, 0))
        hscroll_row.pack_propagate(False)
        self.hscroll = ttk.Scrollbar(hscroll_row, orient="horizontal",
                                     command=self._on_scroll,
                                     style="Fat.Horizontal.TScrollbar")
        self.hscroll.pack(fill="both", expand=True)

        # Meters flanking the waveform - packed last, so it absorbs the slack.
        wave_row = ttk.Frame(timeline, style="Panel.TFrame")
        wave_row.pack(fill="both", expand=True, padx=6)
        self.track_meters = tk.Canvas(wave_row, width=METER_WIDTH,
                                      height=LANE_HEIGHT + RULER_HEIGHT,
                                      background=ui_theme.TIMELINE_BG,
                                      highlightthickness=0)
        self.track_meters.pack(side="left", fill="y")
        ui_theme.attach_tooltip(
            self.track_meters,
            "True-peak meter (peak-hold), measured after this track's own "
            "effects and fader - catches inter-sample overs a plain sample "
            "reading would miss.")
        self.master_meter = tk.Canvas(wave_row, width=METER_WIDTH,
                                      height=LANE_HEIGHT + RULER_HEIGHT,
                                      background=ui_theme.TIMELINE_BG,
                                      highlightthickness=0)
        self.master_meter.pack(side="right", fill="y")
        ui_theme.attach_tooltip(
            self.master_meter,
            "True-peak meter (peak-hold), estimating inter-sample overs a "
            "plain sample reading would miss - the kind a lossy export "
            "re-encode can expose. Measured BEFORE the safety limiter "
            f"engages at {effects.LIMITER_CEILING_DB:.0f} dBTP - so it can "
            "flag a moment the limiter is about to catch, even though what "
            "you actually hear/export stays under that ceiling.")
        # Vertical scrollbar for the lanes. Every speaker gets a LANE_HEIGHT
        # lane, so four tracks need ~315px - more than the timeline pane has at
        # its default size. Without this the pane simply clipped after the
        # first lane and gave no sign the others existed at all.
        #
        # Packed BEFORE the canvas and after the master meter, so it sits
        # between them rather than outside the meter.
        self.wave_vscroll = ttk.Scrollbar(wave_row, orient="vertical",
                                          command=self._on_lane_scroll)
        self.wave_vscroll.pack(side="right", fill="y")
        # Hidden until it is actually needed - see _sync_lane_scroll.
        self.wave_vscroll.pack_forget()

        self.canvas = tk.Canvas(wave_row, height=LANE_HEIGHT + RULER_HEIGHT,
                                background=ui_theme.TIMELINE_BG,
                                highlightthickness=0, cursor="hand2",
                                yscrollcommand=self._on_lane_scrollbar_set)
        self.canvas.pack(side="left", fill="both", expand=True)
        # The waveform is what gives when the pane is resized.
        wave_row.pack_propagate(False)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # Resizing the pane changes how many lanes fit, so the scrollbar has
        # to reappear or vanish with it, not only when tracks are added.
        self.canvas.bind("<Configure>",
                         lambda e: (self._sync_lane_scroll(),
                                    self._draw_waveform()))
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_wheel)

        ttk.Button(zoom, text="-", width=3,
                   command=lambda: self._zoom(1.4)).pack(side="left")
        ttk.Button(zoom, text="+", width=3,
                   command=lambda: self._zoom(1 / 1.4)).pack(side="left", padx=2)
        ttk.Button(zoom, text="Fit", width=5,
                   command=self._zoom_fit).pack(side="left")
        self.zoom_label = ttk.Label(zoom, text="", style="PanelDim.TLabel")
        self.zoom_label.pack(side="left", padx=10)

        # Vertical magnification, separate from the timeline zoom above: the
        # waveform is drawn at true scale, so a quietly-recorded track needs
        # a way to be made readable without pretending it is louder.
        ttk.Label(zoom, text="Height", style="PanelDim.TLabel").pack(side="left")
        ttk.Button(zoom, text="-", width=3,
                   command=lambda: self._zoom_waveform(1 / 2.0)
                   ).pack(side="left", padx=(4, 0))
        ttk.Button(zoom, text="+", width=3,
                   command=lambda: self._zoom_waveform(2.0)).pack(side="left", padx=2)
        ttk.Button(zoom, text="Reset", width=6,
                   command=self._reset_waveform_gain).pack(side="left")
        self.waveform_gain_label = ttk.Label(zoom, text="",
                                             style="PanelDim.TLabel")
        self.waveform_gain_label.pack(side="left", padx=6)

        self.play_hint = ttk.Label(
            zoom, style="PanelDim.TLabel",
            text="click seek   |   drag select   |   shift-drag pan   |   "
                 "wheel zoom   |   shift-wheel height")
        self.play_hint.pack(side="right")

    def _build_vodcast_menu(self, menubar):
        """
        Three recordings of one conversation: V1 host, V2 guest, V3 the merged
        shot with both in frame. V3 is picture only - its audio is the same two
        voices again.
        """
        self.scene_switching = tk.BooleanVar(value=False)
        self.min_shot_seconds = tk.DoubleVar(value=2.0)
        self.max_shot_seconds = tk.DoubleVar(value=25.0)

        menu = tk.Menu(menubar, **ui_theme.menu_options())
        menu.add_command(label="Read me...",
                         command=lambda: self._show_help(
                             "Vodcast", help_text.VODCAST_README))
        menu.add_separator()
        menu.add_command(label="Set merged video (V3)...",
                         command=self.choose_v3)
        self._v3_entry = menu.index("end")
        menu.add_command(label="Clear merged video", command=self.clear_v3)
        menu.add_separator()
        menu.add_checkbutton(label="Switch cameras automatically",
                             variable=self.scene_switching,
                             command=self._on_scene_switching_toggle)
        self._switch_entry = menu.index("end")
        menu.add_command(label="Shot length...", command=self.set_shot_lengths)
        menu.add_command(label="Regenerate camera switching",
                         command=self.regenerate_scenes)
        menu.add_separator()
        menu.add_command(label="Drag along a row in the CAMERAS strip to set "
                               "the camera", state="disabled")
        self.vodcast_menu = menu
        self._add_menu(menubar, "Vodcast", menu)
        self._refresh_vodcast_menu()

    def _refresh_vodcast_menu(self):
        """Keeps the menu honest about what is set and what is possible."""
        import os
        menu = getattr(self, "vodcast_menu", None)
        if menu is None:
            return
        if self.v3_path:
            label = f"✓  V3: {os.path.basename(self.v3_path)}"
        else:
            label = "Set merged video (V3)..."
        try:
            menu.entryconfig(self._v3_entry, label=label)
            # Greyed until it could actually work; the reason is shown when
            # you try, rather than leaving you guessing.
            ready = self.can_switch_cameras() or self.scene_switching.get()
            menu.entryconfig(self._switch_entry,
                             state="normal" if ready else "disabled")
        except Exception:
            pass

    def _build_export_menu(self, menubar):
        """
        Exporting lives on the menu bar rather than a page of its own.

        It is a thing you do at the end, twice, not a place you spend time - it
        was taking a third of the window to hold two buttons and four options.
        The vars are created here because this is now their only home.
        """
        self.export_stems = tk.BooleanVar(value=False)
        self.export_transcript = tk.BooleanVar(value=True)
        self.bake_effects = tk.BooleanVar(value=False)
        self.intro_path = None
        self.outro_path = None

        menu = tk.Menu(menubar, **ui_theme.menu_options())
        menu.add_command(label="Timeline for DaVinci Resolve (FCPXML)...",
                         command=self.export_fcpxml)
        menu.add_command(label="Finished audio (WAV)...",
                         command=self.export_audio_file)
        menu.add_command(label="Finished video (MP4)...",
                         command=self.export_video)
        menu.add_command(label="Transcript only (.srt/.vtt/.txt)...",
                         command=self.export_transcript_only)
        menu.add_separator()
        menu.add_checkbutton(label="Also write one WAV stem per speaker",
                             variable=self.export_stems)
        menu.add_checkbutton(
            label="Write the transcript alongside exports (.srt, .vtt, .txt)",
            variable=self.export_transcript)
        # Without this the FCPXML points at the untouched recordings, and every
        # effect you set up here is simply absent in Resolve.
        menu.add_checkbutton(
            label="Bake effects into the media for Resolve (slower, writes copies)",
            variable=self.bake_effects)
        menu.add_separator()
        # Indices are remembered so the labels can show what is currently set
        # - a menu that never changes gives no way to tell.
        self._bookend_entries = {}
        for which in ("intro", "outro"):
            menu.add_command(
                label=f"Set {which}...",
                command=lambda w=which: self._choose_bookend(w))
            self._bookend_entries[which] = menu.index("end")
            menu.add_command(label=f"Clear {which}",
                             command=lambda w=which: self._clear_bookend(w))
        self._add_menu(menubar, "Export", menu)

        # The two export entries are disabled until there is something to
        # export. Indices 0 and 1, kept here so the enable/disable helper does
        # not have to know the menu's shape.
        self.export_menu = menu
        self._refresh_bookend_labels()
        self.export_menu_entries = (0, 1, 2)
        self._set_export_enabled(False)

    def _refresh_bookend_labels(self):
        """Shows the chosen file beside each bookend entry, with a tick."""
        import os
        for which, index in getattr(self, "_bookend_entries", {}).items():
            path = getattr(self, f"{which}_path", None)
            if path:
                label = f"✓  {which.capitalize()}: {os.path.basename(path)}"
            else:
                label = f"Set {which}..."
            try:
                self.export_menu.entryconfig(index, label=label)
            except Exception:
                pass

    def _set_export_enabled(self, enabled):
        """
        Enables the export entries, minus any that need a picture.

        A WAV or MP3 source is fine for an audio podcast, but there is no
        video to put on a Resolve timeline or encode into an MP4 - so those
        two are left greyed out rather than failing halfway through.
        """
        state = "normal" if enabled else "disabled"
        has_video = any(getattr(m, "has_video", True)
                        for m in (self.speaker_media or []))
        # (0) FCPXML timeline, (1) WAV, (2) MP4 - only the WAV works audio-only.
        needs_video = {0, 2}
        # FCPXML timeline export during camera switching: an earlier attempt
        # described the switch directly in FCPXML and Resolve rearranged the
        # clips it was given - this entry was disabled outright ever since.
        # Fixed 2026-09-13: build_fcpxml now expresses the switch as static-
        # reference picture lanes (enabled/disabled per scene, never
        # swapping which asset a lane points to), confirmed working by
        # direct Resolve import - so FCPXML now follows the same has-video
        # rule as every other export during switching too.
        for index in self.export_menu_entries:
            allowed = state
            if enabled and index in needs_video and not has_video:
                allowed = "disabled"
            self.export_menu.entryconfig(index, state=allowed)

