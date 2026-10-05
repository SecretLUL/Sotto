"""Entry point of Sotto.

Start with:
    pythonw main.py        (no console window)
    python  main.py        (with a console, for troubleshooting)
"""

import sys
import traceback


# The installer of an update: the new version, started by the old one from
# its download folder (audio_transcriber/update.py). It replaces the files
# and has no window of its own.
UPDATE_FLAG = "--apply-update"


def main():
    if UPDATE_FLAG in sys.argv[1:]:
        from audio_transcriber import update
        return update.apply_from_command_line(sys.argv[1:])

    if sys.platform == "win32":
        try:
            import ctypes
            # Enable High-DPI awareness so Windows does not blur or shrink Tkinter on WQHD / 4K monitors
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    try:
        import tkinter as tk
        from audio_transcriber import paths
        from audio_transcriber.ui import chrome
        from audio_transcriber.ui.app import RecorderApp
    except ImportError as exc:

        _fatal(f"A required library is missing: {exc}\n\n"
               f"Install with:\n"
               f"    pip install -r requirements.txt")
        return 1

    try:
        # Before anything reads settings.json or creates bin/ and output/: a
        # packaged build older than this one kept them inside _internal.
        paths.migrate_legacy_data()
        # Before the first window, or the taskbar files it under python.exe.
        chrome.claim_taskbar_identity()
        root = tk.Tk()
        # Here rather than in RecorderApp: the tests build the window too, and
        # keep it invisible.
        chrome.dark_title_bar(root)
        app = RecorderApp(root)
        # Once the window is up: offer to process a recording that was cut
        # short last time. Not in the constructor - the tests build the window
        # too, and a modal question would hang them.
        root.after(400, app.offer_recovery)
        # Later, and in the background: whether a newer version is out.
        root.after(1500, app.start_update_check)
        root.mainloop()
    except Exception:
        _fatal("The application could not be started:\n\n"
               + traceback.format_exc())
        return 1
    return 0


def _fatal(message):
    """Show an error even when the app runs without a console."""
    print(message, file=sys.stderr)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Sotto", message)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
