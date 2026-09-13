"""
Asks what went wrong, immediately before writing the diagnostic report.

A version number and a GPU model tell nobody what broke; the two sentences
someone actually typed are the part that makes a report actionable. Kept as
its own step after the privacy confirmation, not merged into it, so agreeing
to have diagnostics collected and describing the problem stay two separate,
independently skippable things.
"""

import tkinter as tk
from tkinter import ttk

import ui_theme
import version


class ReportDialog(tk.Toplevel):
    """
    Modal. Sets `result` to the typed description (str, possibly empty).

    Never None - closing the window or clicking Skip both mean "no
    description", not "cancel the report"; the report is written either way.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title(f"{version.APP_NAME} - Report a problem")
        self.configure(background=ui_theme.BG)
        self.resizable(True, True)
        self.minsize(440, 320)
        self.transient(parent)
        self.result = ""

        frame = ttk.Frame(self, style="Panel.TFrame")
        frame.pack(fill="both", expand=True, padx=14, pady=12)

        ttk.Label(frame, style="PanelDim.TLabel", justify="left", wraplength=420,
                  text="What were you doing, and what went wrong? Specifics "
                       "help - which track or effect, what you expected to "
                       "happen instead, whether it happens every time. "
                       "Optional, but a report with no description is much "
                       "harder to act on."
                  ).pack(anchor="w", pady=(0, 8))

        text_frame = ttk.Frame(frame, style="Panel.TFrame")
        text_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.text = tk.Text(
            text_frame, width=54, height=10, wrap="word",
            background=ui_theme.PANEL_LIGHT, foreground=ui_theme.TEXT,
            insertbackground=ui_theme.TEXT, relief="flat", borderwidth=0,
            font=ui_theme.FONT, padx=8, pady=8)
        vsb = ttk.Scrollbar(text_frame, orient="vertical",
                            command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        self.text.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        buttons = ttk.Frame(frame, style="Panel.TFrame")
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Save report", width=14,
                   style="Accent.TButton",
                   command=self._accept).pack(side="right")
        ttk.Button(buttons, text="Skip description", width=16,
                   command=self._skip).pack(side="right", padx=(0, 6))

        self.protocol("WM_DELETE_WINDOW", self._skip)
        self.bind("<Escape>", lambda e: self._skip())

        self.update_idletasks()
        self._centre(parent)
        self.grab_set()
        self.text.focus_set()

    def _centre(self, parent):
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _accept(self):
        self.result = self.text.get("1.0", "end").strip()
        self.destroy()

    def _skip(self):
        self.result = ""
        self.destroy()


def ask(parent):
    """Shows the dialog and returns the typed description (possibly "")."""
    dialog = ReportDialog(parent)
    parent.wait_window(dialog)
    return dialog.result
