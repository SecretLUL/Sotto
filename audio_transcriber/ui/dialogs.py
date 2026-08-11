"""Modal dialogs that need more than tkinter.messagebox can express.

messagebox offers fixed button sets only (yes/no/cancel). The overwrite
warning has three genuinely different outcomes - overwrite, number, rename -
plus an entry field, so it is drawn here in the style of the rest of the
interface.
"""

import tkinter as tk
from tkinter import ttk

from .. import paths
from . import icons
from . import theme as T
from . import widgets as W

MIN_WIDTH = 620


def ask_output_conflict(parent, out_dir, base_name):
    """Warn that '<base_name>.wav' / '.txt' already exist in out_dir.

    Returns the base name the run should use - the original one (overwrite),
    a numbered one or a freely chosen one - or None when the user cancelled.
    """
    dialog = OutputConflictDialog(parent, out_dir, base_name)
    parent.wait_window(dialog)
    return dialog.result


class OutputConflictDialog(tk.Toplevel):
    """Asks whether to overwrite, number or rename the output files."""

    def __init__(self, parent, out_dir, base_name):
        super().__init__(parent, bg=T.BG)
        self.out_dir = out_dir
        self.base_name = base_name
        self.suggestion = base_name
        self.result = None

        self.title("File already exists")
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _event: self._cancel())

        self._build()
        self._show(base_name)
        self._place_over(parent)

        self._take_grab()
        self.entry.focus_set()

    def _take_grab(self, attempts=20):
        """Go modal as soon as the window is actually on screen.

        grab_set() fails while the window is not viewable yet. wait_visibility()
        is not an option: Tk withdraws a transient window together with its
        master, so it would block forever whenever the dialog is opened from a
        hidden parent (the GUI tests do exactly that).
        """
        try:
            self.grab_set()
        except tk.TclError:
            try:
                if attempts > 0:
                    self.after(80, lambda: self._take_grab(attempts - 1))
            except tk.TclError:          # pragma: no cover - dialog already gone
                pass

    # ------------------------------------------------------------------
    def _build(self):
        frame = tk.Frame(self, bg=T.BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=T.XL, pady=T.LG)

        head = tk.Frame(frame, bg=T.BG)
        head.pack(fill=tk.X)
        self._icon = icons.get_icon("warning", size=30)
        tk.Label(head, image=self._icon, bg=T.BG).pack(side=tk.LEFT, padx=(0, T.SM))
        tk.Label(head, text="This file already exists", bg=T.BG, fg=T.TEXT,
                 font=T.fonts["title"], anchor="w").pack(side=tk.LEFT)

        wrap = MIN_WIDTH - 2 * T.XL
        self.message = tk.Label(frame, bg=T.BG, fg=T.TEXT_DIM,
                                font=T.fonts["small"], anchor="w",
                                justify="left", wraplength=wrap)
        self.message.pack(fill=tk.X, pady=(T.MD, 0))

        self.folder = tk.Label(frame, text=self.out_dir, bg=T.BG, fg=T.TEXT_MUTE,
                               font=T.fonts["mono_small"], anchor="w",
                               justify="left", wraplength=wrap)
        self.folder.pack(fill=tk.X, pady=(T.XS, 0))

        name_field = W.Field(
            frame, "File name",
            lambda parent: ttk.Entry(parent, style="Dark.TEntry",
                                     font=T.fonts["body"]),
            bg=T.BG)
        name_field.pack(fill=tk.X, pady=(T.MD, 0))
        self.entry = name_field.widget
        self.entry.bind("<Return>", lambda _event: self._rename())
        self.entry.bind("<KeyRelease>", lambda _event: self._check_entry())

        self.hint = tk.Label(frame, bg=T.BG, fg=T.TEXT_MUTE, font=T.fonts["tiny"],
                             anchor="w", justify="left")
        self.hint.pack(fill=tk.X, pady=(T.XS, 0))

        row = tk.Frame(frame, bg=T.BG)
        row.pack(fill=tk.X, pady=(T.LG, 0))

        self.rename_btn = self._button(row, "Save under this name", "accent",
                                       self._rename)
        self.rename_btn.pack(side=tk.RIGHT)
        self.number_btn = self._button(row, "Number it", "ghost",
                                       self._auto_number)
        self.number_btn.pack(side=tk.RIGHT, padx=(0, T.SM))
        self.overwrite_btn = self._button(row, "Overwrite", "record",
                                          self._overwrite)
        self.overwrite_btn.pack(side=tk.RIGHT, padx=(0, T.SM))
        self.cancel_btn = self._button(row, "Cancel", "quiet", self._cancel)
        self.cancel_btn.pack(side=tk.LEFT)

    @staticmethod
    def _button(parent, text, kind, command):
        return W.Button(parent, text=text, kind=kind, bg=T.BG, height=38,
                        width=_button_width(text), command=command)

    # ------------------------------------------------------------------
    def _show(self, name):
        """Point the dialog at `name`: message, suggestion and entry."""
        self.base_name = name
        self.suggestion = paths.next_free_name(self.out_dir, name)

        conflicts = paths.existing_outputs(self.out_dir, name)
        verb = "exists" if len(conflicts) == 1 else "exist"
        self.message.configure(
            text=f"{_join(conflicts)} already {verb} in the output folder. "
                 f"Recording under this name overwrites the file"
                 f"{'' if len(conflicts) == 1 else 's'} without a further "
                 f"warning.")

        self.entry.delete(0, tk.END)
        self.entry.insert(0, name)
        self.entry.selection_range(0, tk.END)

        label = f'Number it: {self.suggestion}'
        self.number_btn.configure(text=label, width=_button_width(label))
        self._check_entry()

    def _check_entry(self):
        """Live feedback on whether the typed name is still free."""
        name = paths.safe_output_name(self.entry.get(), default=self.base_name)
        conflicts = paths.existing_outputs(self.out_dir, name)
        if conflicts:
            self.hint.configure(text=f"⚠ {_join(conflicts)} would be replaced.",
                                fg=T.WARN)
        else:
            self.hint.configure(text=f"✓ '{name}' is still free.", fg=T.OK)

    # ------------------------------------------------------------------
    def _overwrite(self):
        self._finish(self.base_name)

    def _auto_number(self):
        self._finish(self.suggestion)

    def _rename(self):
        # An empty field keeps the current name instead of silently falling
        # back to the 'my_meeting' default of safe_output_name().
        name = paths.safe_output_name(self.entry.get(), default=self.base_name)
        if paths.existing_outputs(self.out_dir, name):
            # Still taken: stay open and warn about the new name instead.
            self.bell()
            self._show(name)
            return
        self._finish(name)

    def _cancel(self):
        self._finish(None)

    def _finish(self, result):
        self.result = result
        try:
            self.grab_release()
        except tk.TclError:              # pragma: no cover - no window manager
            pass
        self.destroy()

    # ------------------------------------------------------------------
    def _place_over(self, parent):
        """Centre the dialog on the main window, slightly above the middle."""
        self.update_idletasks()
        width = max(MIN_WIDTH, self.winfo_reqwidth())
        height = self.winfo_reqheight()
        try:
            x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - height) // 3
            self.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
        except tk.TclError:              # pragma: no cover - no window manager
            self.geometry(f"{width}x{height}")


# ----------------------------------------------------------------------
def _button_width(text):
    return T.fonts["button"].measure(text) + 40


def _join(names):
    """'a', 'a and b', 'a, b and c' - quoted for the message text."""
    quoted = [f"'{name}'" for name in names]
    if not quoted:
        return "The output file"
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"
