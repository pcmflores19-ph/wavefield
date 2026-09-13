"""
Picks a reference track, then reviews (and lets you edit) the detected sync
offset for every other track before anything actually renders.

Detection itself (sync.find_offset) is pure numpy and fast even for a long
episode, so it runs synchronously here, right after a reference is chosen -
no busy indicator needed for this part. The slow part (actually cutting or
padding the audio/video) only happens after this dialog returns, back in
app.py's threaded worker.
"""

import os
import tkinter as tk
from tkinter import ttk

import sync
import ui_theme
import version


class SyncDialog(tk.Toplevel):
    """Modal. Sets `result` to {index: offset_seconds}, or None if cancelled."""

    def __init__(self, parent, speaker_paths, per_speaker_speech, speech_levels,
                 timeline_duration):
        super().__init__(parent)
        self.title(f"{version.APP_NAME} - Sync")
        self.configure(background=ui_theme.BG)
        self.resizable(False, False)
        self.transient(parent)
        self.result = None

        self.speaker_paths = speaker_paths
        self.per_speaker_speech = per_speaker_speech
        self.speech_levels = speech_levels
        self.timeline_duration = timeline_duration
        self._offset_vars = {}   # index -> tk.StringVar
        self._status_labels = {}  # index -> widget
        self._manual_vars = {}   # index -> tk.BooleanVar
        self._entries = {}       # index -> ttk.Entry

        frame = ttk.Frame(self, style="Panel.TFrame")
        frame.pack(fill="both", expand=True, padx=14, pady=12)

        ttk.Label(frame, style="PanelDim.TLabel", justify="left", wraplength=460,
                  text="Aligns every other recording to the one you pick here, "
                       "by matching either when each person talks or - if "
                       "they were recorded in the same room - the actual "
                       "sound each mic picked up. Each row below shows what "
                       "was detected automatically; tick \"Manual\" on a row "
                       "to type your own offset instead - the only way this "
                       "changes anything is if you tick it and enter a value."
                  ).pack(anchor="w", pady=(0, 10))

        ttk.Label(frame, text="Align to", style="Panel.TLabel").pack(anchor="w")
        self.reference_box = ttk.Combobox(
            frame, state="readonly", width=50,
            values=[os.path.basename(p) for p in speaker_paths])
        self.reference_box.current(0)
        self.reference_box.pack(fill="x", pady=(2, 10))
        self.reference_box.bind("<<ComboboxSelected>>", lambda e: self._detect())

        self.rows_frame = ttk.Frame(frame, style="Panel.TFrame")
        self.rows_frame.pack(fill="both", expand=True)

        buttons = ttk.Frame(frame, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Sync", width=12, style="Accent.TButton",
                   command=self._accept).pack(side="right")
        ttk.Button(buttons, text="Cancel", width=12,
                   command=self.destroy).pack(side="right", padx=(0, 6))

        self.bind("<Escape>", lambda e: self.destroy())
        self._detect()

        self.update_idletasks()
        self._centre(parent)
        self.grab_set()

    def _centre(self, parent):
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _reference_index(self):
        return self.reference_box.current()

    def _detect(self):
        """Rebuilds the offset-review rows for the currently chosen reference."""
        for child in self.rows_frame.winfo_children():
            child.destroy()
        self._offset_vars.clear()
        self._status_labels.clear()
        self._manual_vars.clear()
        self._entries.clear()

        reference_index = self._reference_index()
        reference = sync.TrackActivity(
            self.per_speaker_speech[reference_index], self.timeline_duration,
            self._levels_for(reference_index))

        for index, path in enumerate(self.speaker_paths):
            if index == reference_index:
                continue
            target = sync.TrackActivity(
                self.per_speaker_speech[index], self.timeline_duration,
                self._levels_for(index))
            result = sync.find_offset(reference, target)
            self._add_row(index, path, result)

    def _levels_for(self, index):
        levels = (self.speech_levels[index]
                  if self.speech_levels and index < len(self.speech_levels) else None)
        return levels if levels is not None and len(levels) else None

    def _add_row(self, index, path, result):
        row = ttk.Frame(self.rows_frame, style="Panel.TFrame")
        row.pack(fill="x", pady=(0, 4))

        ttk.Label(row, text=os.path.basename(path), style="Panel.TLabel",
                  width=24).pack(side="left")

        # Manual starts ticked (and the field already editable) exactly when
        # detection couldn't confidently place a value - there's nothing
        # useful to lock in that case. Otherwise it starts unticked: the
        # detected value is shown but not typable until you deliberately
        # opt in, so a stray click can't quietly change a good result.
        manual_var = tk.BooleanVar(value=result.ambiguous)
        self._manual_vars[index] = manual_var

        offset_var = tk.StringVar(value=f"{result.offset_seconds:+.3f}")
        entry = ttk.Entry(row, textvariable=offset_var, width=10, justify="right",
                          state="normal" if result.ambiguous else "readonly")
        entry.pack(side="left", padx=(4, 4))
        ttk.Label(row, text="s", style="PanelDim.TLabel").pack(side="left")
        self._offset_vars[index] = offset_var
        self._entries[index] = entry

        def on_toggle(entry=entry, manual_var=manual_var):
            entry.configure(state="normal" if manual_var.get() else "readonly")

        ttk.Checkbutton(row, text="Manual", style="Toggle.TCheckbutton",
                        variable=manual_var, command=on_toggle
                        ).pack(side="left", padx=(6, 0))

        status = tk.Label(row, background=ui_theme.PANEL, anchor="w")
        status.pack(side="left", padx=(10, 0), fill="x", expand=True)
        self._set_status(status, result)
        self._status_labels[index] = status

    def _set_status(self, label, result):
        if result.ambiguous:
            label.configure(text="not confident - set manually",
                            fg=ui_theme.WARN_YELLOW)
            return
        method_name = "shared audio" if result.method == "content" else "speech timing"
        # A plain-language tier instead of the raw score: the number itself
        # (an unbounded, median-relative measure) means nothing to look at,
        # only to threshold against - anything shown at all already cleared
        # sync.MIN_CONFIDENCE, so the only useful thing left to communicate
        # is roughly how much it cleared it by.
        tier = "strong match" if result.confidence >= sync.MIN_CONFIDENCE * 2 \
            else "plausible match"
        label.configure(text=f"detected via {method_name} - {tier}",
                        fg=ui_theme.TEXT_DIM)

    def _accept(self):
        offsets = {}
        for index, var in self._offset_vars.items():
            try:
                offsets[index] = float(var.get())
            except ValueError:
                self._status_labels[index].configure(
                    text="not a number - fix or leave 0", fg=ui_theme.WARN_YELLOW)
                return
        self.result = offsets
        self.destroy()


def ask(parent, speaker_paths, per_speaker_speech, speech_levels, timeline_duration):
    """
    Shows the dialog and returns {index: offset_seconds} for every track
    except the chosen reference, or None if cancelled.
    """
    dialog = SyncDialog(parent, speaker_paths, per_speaker_speech, speech_levels,
                        timeline_duration)
    parent.wait_window(dialog)
    return dialog.result
